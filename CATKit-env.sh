#!/bin/bash

echo "Setting CATKit Environmant Variables..."
export CATKIT_BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export CATKIT_DATA=${CATKIT_BASEDIR}/data
export CATKIT_SDK=${CATKIT_BASEDIR}/sdk
export CATKIT_EXT=${CATKIT_BASEDIR}/external

echo "  >> CATKIT_BASEDIR => ${CATKIT_BASEDIR}"
echo "    >> CATKIT_DATA  => ${CATKIT_DATA}"
echo "    >> CATKIT_SDK   => ${CATKIT_SDK}"
echo "    >> CATKIT_EXT   => ${CATKIT_EXT}"