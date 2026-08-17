## [ERR-20260817-001] jekyll-build-missing-git-gem

**Logged**: 2026-08-17T10:26:59+08:00
**Priority**: medium
**Status**: resolved
**Area**: tests

### Summary
`bundle exec jekyll build` cannot start when the worktree-local `jekyll-terser` Git gem checkout is missing.

### Error
`Bundler::GitError: https://github.com/RobertoJBeltran/jekyll-terser.git is not yet checked out. Run bundle install first.`

### Context
- Command: `bundle exec jekyll build`
- Worktree: `/data/share/mpaper_sandbox/runs/agifrontier_seo_batch_100_20260816/publish_worktree`
- Missing path: `vendor/bundle/ruby/3.0.0/bundler/gems/jekyll-terser-1085bf66d692`

### Suggested Fix
Run `bundle install --jobs 4 --retry 3` in the worktree and wait for the explicit `Bundle complete!` message before rerunning the build.

### Metadata
- Reproducible: yes
- Related Files: Gemfile, Gemfile.lock
## [ERR-20260817-001] playwright-computed-style-selector

**Logged**: 2026-08-17T12:20:00+08:00
**Priority**: medium
**Status**: resolved
**Area**: frontend

### Summary
The first article-table selector did not match tutorials rendered with the default layout.

### Error
`TypeError: Failed to execute 'getComputedStyle' on 'Window': parameter 1 is not of type 'Element'.`

### Context
- Computed-style verification queried `#markdown-content table`.
- Tutorials explicitly use `layout: default`, so their tables are direct children of `.container[role="main"]`.

### Suggested Fix
Inspect the built DOM before finalizing layout-scoped selectors and cover both post-layout and default-layout article containers.

### Metadata
- Reproducible: yes
- Related Files: _sass/_base.scss, _layouts/default.liquid, _layouts/post.liquid

## [ERR-20260817-002] zsh-readonly-status-variable

**Logged**: 2026-08-17T21:24:00+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
Using `status` as a temporary exit-code variable breaks verification commands under zsh.

### Error
`zsh: read-only variable: status`

### Context
- Command captured a validator exit code with `status=$?`.
- Repository commands run under zsh, where `status` is a special read-only parameter.

### Suggested Fix
Use a neutral name such as `exit_code=$?` when preserving command exit status.

### Metadata
- Reproducible: yes
- Related Files: scripts/check_built_site.py

## [ERR-20260817-003] parallel-command-directory-race

**Logged**: 2026-08-17T21:37:00+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
Commands that create and consume the same directory cannot run in one parallel batch.

### Error
`Failed to create unified exec process: No such file or directory`

### Context
- One command created the Playwright artifact directory.
- A parallel command used that directory as its working directory before creation completed.

### Suggested Fix
Create required directories first, then run independent browser commands in parallel only after the prerequisite succeeds.

### Metadata
- Reproducible: yes
- Related Files: none
