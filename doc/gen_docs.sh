#!/bin/bash
sphinx-apidoc --ext-autodoc -o doc ./sparkly/
pushd doc
make html
