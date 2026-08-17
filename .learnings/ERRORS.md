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
