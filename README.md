# pyppca
Probabilistic PCA which is applicable also on data with missing values. Missing value estimation is typically better than NIPALS but also slower to compute and uses more memory. [A port to Python of the implementation by Jakob Verbeek](http://lear.inrialpes.fr/~verbeek/software.php).

Usage:
```
from pyppca import ppca
C, ss, M, X, Ye = ppca(Y,d,dia)
```
