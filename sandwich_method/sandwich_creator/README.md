# SANDOR

SANDOR was created for construction sandwich-structures from liquid and solid systems. You can use [this repository](https://github.com/AlexanderKvashnin/HEM) for more details.

**Required libraries:**
- numpy==1.26.4
- pandas==2.0.3
- ase==3.22.1

**Example of application:**

```
from sandor.functions import get_sandwich_structures

path_solid = '<path/to/solid/structure>'
path_liquid = '<path/to/liquid/structure>'

sandwich_structures = get_sandwich_structures(path_solid, path_liquid)

#sandwich_structures -> list[ase.atoms.Atoms]
```