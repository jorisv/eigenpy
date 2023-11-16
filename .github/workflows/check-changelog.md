name: Check-changelog
on:
  pull_request:
    branches:
      - topic/test
jobs:
  check-changelog:
    name: Check changelog action
    runs-on: ubuntu-latest
    steps:
      - uses: tarides/changelog-check-action@v2
        with:
          changelog: CHANGELOG.md
