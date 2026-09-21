#! /usr/bin/env bash

### cherry pick statuses
OK=0
SKIP=10
QUIT=20
CONFLICT=30

# Shorthand for git rev-parse
parse() { git rev-parse "$@"; }

# Shorthand for git show
show() { git show "$@"; }

# Hard reset
hard_reset() { git reset --hard "$@"; }

# List all mainline commits on the branch
mainline_commits() { git rev-list --first-parent "$@"; }

# Return the number of mainline commits on the branch
count() { mainline_commits --count "$@"; }

# Shorthand for git cherry-pick
# DO NOT remove `-x` from this command, it is what allows
# us to find the original commits.
pluck() { git cherry-pick -x "$@"; }

# Get a human-readable summary of a commit given a SHA
summarize_commit() { git --no-pager show -s --format='Author: %an <%ae>%nDate:   %ad%n%n%s%n%n%b' --date=short "$1"; }

# Check if $1 is an ancestor of $2.
is_ancestor() { git merge-base --is-ancestor "$1" "$2"; }

# Determine if a commit is a merge commit (i.e. it has a second parent)
is_merge_commit() { parse --verify "$1^2" >/dev/null 2>&1; }

to_tree() { local commit="$1"; parse "${commit}^{tree}"; }

# Confirm the current branch is `public`
current_branch_is_public() {
    local current
    current=$(git rev-parse --abbrev-ref HEAD)
    if [[ "$current" != "public" ]]; then
        echo "Currently on branch <$current> - this script can only be run from <public>."
        exit 1
    fi
}

# Confirm there are no modified/untracked files in working tree.
clean_working_tree() {
    if [[ -n $(git status --porcelain) ]]; then
        echo "Working tree is not clean:"
        git status --porcelain
        exit 1
    fi
}

require_args() {
  local expected="$1" usage="$2"
  shift 2
  if [[ "$#" -ne "$expected" ]]; then
    printf "%s\n" "$usage" >&2
    return 2
  fi
}

# Given a cherry pick commit, get the original commit SHA.
extract_original_sha_from_cherry_pick() {
    require_args 1 "Usage: extract_original_sha_from_cherry_pick <commit sha>" "$@" || return;
	local original_sha
	original_sha=$(
		show "$1" -s --format=%B |
			grep 'cherry picked from commit' |
			awk '{print $NF}' |
			tr -d ')'
	)

	[[ -n "$original_sha" ]] || return 1
	echo "$original_sha"
}

# Defensive guard to confirm that a SHA actually exists on a given branch.
confirm_sha_on_branch() {
    require_args 2 "Usage: confirm_sha_on_branch <commit sha> <branch>" "$@" || return;
    local sha="$1" source="$2"
    is_ancestor "$sha" "$source" || {
        printf "Commit %s is not on branch %s.\n" "$sha" "$source" >&2
        return 1
    }
}


# Cherry pick an individual commit interactively.
# Offers to either accept the commit as-is, edit the message,
# or skip entirely. 
# NOTE Skipping a commit will cause divergent commit-trees.
cherry_pick_interactive() {
    require_args 1 "Usage: cherry_pick_interactive <commit_sha>" "$@" || return;

	local sha="$1"
	while true; do
	    summarize_commit "$sha"
	    printf "Action: Accept as-is (Enter, default); [e]dit commit message, [s]kip, [q]uit: "
		read -r choice
		choice="${choice:-a}"

		local -a options=()

		[[ "$choice" =~ ^[eE]$ ]] && options+=(-e)

		is_merge_commit "$sha" && options+=(-m 1)

		case "$choice" in
		e | E | a | A)
			if ! pluck "${options[@]}" "$sha"; then
				echo "Cherry-pick hit a conflict."
				return "$CONFLICT"
			fi
			return "$OK"
			;;
		s | S)
			echo "Skipped ${sha}"
			return "$SKIP"
			;;
		q | Q)
			echo "Stopped by user."
			return "$QUIT"
			;;
		*)
			echo "Invalid choice."
			;;
		esac
	done
}
