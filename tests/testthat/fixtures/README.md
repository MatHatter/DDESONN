adagrad-x-baseline.rds was generated using the unmodified installed DDESONN
package at commit 5dd541a7fe5c6fc56ddc5b7d4a3e8b0ec7d55373.

It records three sequential updates for 16 configurations: single/multilayer,
no regularization plus L1, L2, L1_L2, configured Group_Lasso and Max_Norm, and
individual weight/bias update flags. Each snapshot includes actual parameter
values and shapes, all optimizer state, groups, and the final RNG state.

Two three-epoch end-to-end regression runs additionally record parameters,
predictions, and RNG state, including multilayer dropout. Optimizer is omitted
so the fixture captures the original default and tests the new default.

These fixtures preserve the original implementation's behavior; they are not
reference results for standard mathematical Adagrad. Do not regenerate them
from adagrad_x or the corrected optimizer code.
