#! /bin/bash

find include/ -iname *.h -o -iname *.cpp | xargs clang-format -i
find src/ -iname *.h -o -iname *.cpp | xargs clang-format -i
