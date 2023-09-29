# PROVIDI instructions

Here we describe how to run experiments on the PROVIDI server.

## Git worktrees

We recommend using git worktrees for running long experiments or for running a lenghty test suite. The reason why they
are useful for simulation experiments is that during simulation you want to read or write to the disk and git worktrees
allow you to have the project checked out in a branch while you work on other stuff in a different directory.

More information on git worktrees can be found at https://git-scm.com/docs/git-worktree

## Server configuration using pycharm proffesional

If you are using pycharm professional (free for students) you can setup an SSH interpreter configuration, this will
allow you to automatically syncronize your local files to the PROVIDI server through the SSH protocol and execute the
code using an interpreter on the server. So you might have to setup the python environment on the server but pycharm
will prompt you if you follow the instructions
on https://www.jetbrains.com/help/pycharm/configuring-remote-interpreters-via-ssh.html#ssh


