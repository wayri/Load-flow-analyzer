# Acknowledgements

Load flow analyzer was originally developed by Yawar Badri in 2019.

The project draws on established power-system analysis methods and the following
technical references and software ecosystems:

- [MATPOWER](https://matpower.org/): AC power-flow formulations and the case9
  network used in numerical tests.
  The [IEEE benchmark collection](benchmarks/README.md) retains pinned case-file
  headers, upstream usage notes and [source provenance](benchmarks/sources/matpower/provenance.json).
- [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/): array operations,
  sparse matrices and numerical solvers used by NRlarge.
- [Matplotlib](https://matplotlib.org/), [NetworkX](https://networkx.org/) and
  [Pillow](https://python-pillow.org/): benchmark plots, network layouts and image previews.
- [PETSc](https://petsc.org/): distributed-network and scalable-solver references.
- [Khronos Group](https://www.khronos.org/): OpenCL and OpenGL specifications.
- [PyOpenGL](https://pyopengl.sourceforge.net/) and [GLFW](https://www.glfw.org/):
  OpenGL bindings and context support where used.

These references and dependencies retain their respective ownership and licenses.
Acknowledgement does not imply affiliation, sponsorship or endorsement.

## Disclaimer and waiver

The software, source code, documentation, examples, datasets and generated results
are provided **"AS IS" and "AS AVAILABLE"**, without warranties or representations
of any kind, whether express, implied or statutory. This includes warranties of
accuracy, completeness, reliability, merchantability, fitness for a particular
purpose and non-infringement.

The software and its results may contain errors, omissions, defects or numerical
inaccuracies. Users are responsible for reviewing input data, assumptions, models
and outputs; obtaining independent validation appropriate to their application;
and deciding whether the software is suitable for their intended use.

To the fullest extent permitted by applicable law, use is at the user's own risk.
The authors, contributors and copyright holders disclaim liability for any loss,
damage, injury, claim or expense arising from or related to use of, inability to
use, or reliance on the software, documentation or results. This includes direct,
indirect, incidental, consequential, special and punitive damages, loss of data,
lost profits, business interruption and equipment or system damage, regardless
of the legal basis of the claim or whether such harm was foreseeable.

To the fullest extent permitted by applicable law, users waive claims against the
authors, contributors and copyright holders arising from such use or reliance.
Nothing in this notice excludes liability or waives rights that applicable law
does not permit to be excluded or waived.
