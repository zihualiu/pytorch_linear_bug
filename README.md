# pytorch_linear_bug
PyTorch bug where linear layer *sometimes* outputs NaN

Give it a few runs because the bug is very inconsistent.
I tested on python 3.5.2 and pytorch 0.3.0.post4

------------------------- EDIT -------------------------
The issue disappeared after I rebuilt a new virtual env. I suspect there are some conflicting libraries under the hood with some other libraries I'm using.
