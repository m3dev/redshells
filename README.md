# redshells

[![Test](https://github.com/m3dev/redshells/actions/workflows/test.yml/badge.svg)](https://github.com/m3dev/redshells/actions/workflows/test.yml)
[![Python Versions](https://img.shields.io/pypi/pyversions/redshells.svg)](https://pypi.org/project/redshells/)
[![](https://img.shields.io/pypi/v/redshells)](https://pypi.org/project/redshells/)
![](https://img.shields.io/pypi/l/redshells)

Machine learning tasks which are used with data pipeline library "luigi" and its wrapper "gokart".

## Dependencies

You should add `tensorflow = ">=1.13.1,<2.0"` on your app dependencies if you would like to use some models under `redshells/contrib` directory.
`Tensorflow` is not included redshells' dependencies because there are some models not used it.
