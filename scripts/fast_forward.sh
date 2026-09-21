#!/usr/bin/env bash

# Fast forward the target branch by cherry picking commits from `main`.
# First, this function checks that the tip of HEAD is a cherry picked commit done with -x.
# If it is not, then either
#   a) The branch is a brand-new orphan, or
#   b) Someone manually cherry-picked without `-x`
# In both scenarios, you will have to traverse commits on `main` to determine when
# the commit-trees are equal, and cherry pick everything after that.

# If it is a cherry pick commit, it extracts the original SHA of the commit and confirms
# that that SHA exists on `main`.

# If that is the most recent SHA on `main`, it exits as there are no more updates on `main`.

# Otherwise, it traverses each commit from that SHA to main:HEAD and asks the user if
# they would like to:
#   a) Accept the commit as-is
#   b) Edit the commit message
#   c) Skip the commit (WARNING: Creates divergent commit-trees)
#   d) Quit (hard reset; undoes any previous cherry picks in this session)
fast_forward() {

	local target="$1"
	local source="main"

    local starting_head
    starting_head=$(parse HEAD)

	local original_sha
	original_sha=$(extract_original_sha_from_cherry_pick "$target") || {
		echo "Latest commit on ${target} was not cherry picked from ${source}" >&2
		return 1
	}

    confirm_sha_on_branch "$original_sha" "$source" || {
        echo "Refusing to sync: original SHA is not on source branch '$source'." >&2
        return 1
    }

    if [[ "$original_sha" == $(parse "$source") ]]; then
        echo "Everything up to date!"
        echo "No further commits on ${source} to add onto ${target}."
        return 0
    fi


    local idx
    idx=1

    local num_commits
    num_commits=$(count "$original_sha..$source")

    local commits=()
    while IFS= read -r commit; do
        commits+=("$commit")
    done < <(mainline_commits --reverse "$original_sha..$source")

    for commit in "${commits[@]}"; do
        echo "[$idx/$num_commits]............................................"
        cherry_pick_interactive "$commit"; local return_code=$?

        case "$return_code" in
            "$OK"|"$SKIP") ;;
            "$QUIT")
                echo "Hard resetting back to before this process started."
                hard_reset "$starting_head"
                return 1
                ;;
            "$CONFLICT")
                echo "Resolve conflict then run git cherry-pick --continue or --abort" >&2
                return 1
                ;;
            *)
                echo "Unexpected cherry-pick status: $return_code" >&2
                return 1
                ;;
        esac

        idx=$((idx + 1))

        
    done
}
