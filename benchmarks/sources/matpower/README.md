# Pinned MATPOWER-format IEEE case data

The five `.m` files are unchanged copies from MATPOWER revision
`80b6e16f760f58721ad31221c7287c28259cd4ad`, retrieved on 10 November 2026.
[provenance.json](provenance.json) records the exact source URLs, byte counts and
SHA-256 hashes. Original headers, conversion notes and warnings are retained.

| File | Network |
| --- | --- |
| `case14.m` | IEEE 14-bus |
| `case_ieee30.m` | IEEE 30-bus archive conversion |
| `case57.m` | IEEE 57-bus |
| `case118.m` | IEEE 118-bus |
| `case300.m` | IEEE 300-bus, including the upstream nominal-tap updates |

The supplied [MATPOWER LICENSE](LICENSE) expressly distinguishes MATPOWER code
from the distributed case data; its BSD code license is not assigned to these
case files by this repository. The case headers identify the
[UW Power Systems Test Case Archive](https://labs.ece.uw.edu/pstca/).
The archive's [content policy](https://labs.ece.uw.edu/pstca/pg_tcaintro.htm)
describes research and educational use. These files support the research and
educational benchmark studies in this collection. Retain their provenance and
check the source usage conditions when reusing them.

Local `README.md` and `provenance.json` describe the import; they are separate
from the unchanged upstream files. The importer reads literal numeric data and
does not execute MATLAB/Octave code.
