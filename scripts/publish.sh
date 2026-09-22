#! /usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
# shellcheck source=fast_forward.sh
source "$SCRIPT_DIR/fast_forward.sh"

current_branch_is_public
clean_working_tree

fast_forward public || exit 1

echo "The following files have been edited and committed:"
echo "------------------"
git diff origin/public --name-only
echo "------------------"
echo "Run 'git push' when ready to publish."