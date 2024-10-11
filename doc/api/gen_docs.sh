#!/bin/bash
sphinx-apidoc --ext-autodoc -o . ../../sparkly/
# pushd doc/api
make html
