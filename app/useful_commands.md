From the repo root:
```
find app -type f -name '*.pyc' -delete
```
That walks everything under app/ and deletes only regular files named \*.pyc.

To preview first:
```
find app -type f -name '*.pyc' -print
```
To also remove empty __pycache__ dirs (optional):
```
find app -type d -name '__pycache__' -exec rm -rf {} +
```
Use that second line only if you intend to drop cached bytecode directories entirely (usually fine before a clean commit or packaging).