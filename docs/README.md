## Prerequisites

The documentation building is done using sphinx. You can find installation instructions
at https://www.sphinx-doc.org/en/master/usage/installation.html

## Building docs

To generate the documentation run `make html` from this directory. The resulting documentation will be in
build/index.html. I personally recommend running the following command

```
make html -v
```

This will ensure that you get all the output errors and warnings of the build allowing you to correct style errors.

## Generating package rst files

To generate the rst file for a package you can use the following cli
tool https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html.

I recommend using the following flags as well for better understanding of how the tool is building

```
sphinx-apidocs -M -o ./build/microtool_api
```

By using the `-M` flag it makes sure that internal references put the module before submodules such that the hierarchy
in the documentation makes more sense.

