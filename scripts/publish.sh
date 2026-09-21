#! /usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
# shellcheck source=fast_forward.sh
source "$SCRIPT_DIR/fast_forward.sh"

current_branch_is_public
clean_working_tree


datetime=$(date "+%y-%m-%d-%H%M")
new_branch_name="update_public_$datetime"
git switch -c "$new_branch_name"

fast_forward "$new_branch_name"

git push \
  -o merge_request.create \
  -o merge_request.target=public \
  -o merge_request.remove_source_branch \
  -o merge_request.title="<Your title here!>"
  origin "$new_branch_name"