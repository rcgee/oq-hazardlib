#!/bin/bash
#
# makerpm.sh  Copyright (c) 2015, GEM Foundation.
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake.  If not, see <http://www.gnu.org/licenses/>

# Work in progress

set -e

CUR=$(pwd)
BASE=$(cd $(dirname $0)/.. && /bin/pwd)

REPO=oq-hazardlib
BRANCH='HEAD'
EXTRA=''

while (( "$#" )); do
    case "$1" in
        "-h")
            echo "Usage: $0 [-c] [-l] [BRANCH]"
            echo -e "\nOptions:\n\t-l: build RPM locally\n\t-c: clean build dir"
            exit 0
            ;;
        "-l")
            BUILD=1
            shift
            ;;
        "-c")
            rm -Rf $BASE/build-rpm
            echo "$BASE/build-rpm cleaned"
            exit 0
            ;;
        *)
            BRANCH="$1"
            shift
            ;;
    esac
done

cd $BASE
mkdir -p build-rpm/{RPMS,SOURCES,SPECS,SRPMS}

LIB=$(cut -d "-" -f 2 <<< $REPO)
SHA=$(git rev-parse --short $BRANCH)
VER=$(cat openquake/${LIB}/__init__.py | sed -n "s/^__version__[  ]*=[    ]*['\"]\([^'\"]\+\)['\"].*/\1/gp")
TIME=$(date +"%s")
echo "$LIB - $BRANCH - $SHA - $VER"

sed "s/##_repo_##/${REPO}/g;s/##_version_##/${VER}/g;s/##_release_##/git${SHA}/g;s/##_timestamp_##/${TIME}/g" rpm/python-${REPO}.spec.inc > build-rpm/SPECS/python-${REPO}.spec

git archive --format=tar --prefix=${REPO}-${VER}-git${SHA}/ $BRANCH | gzip -9 > build-rpm/SOURCES/${REPO}-${VER}-git${SHA}.tar.gz

mock -r openquake --buildsrpm --spec build-rpm/SPECS/python-${REPO}.spec --source build-rpm/SOURCES --resultdir=build-rpm/SRPMS/
if [ "$BUILD" == "1" ]; then
    mock -r openquake build-rpm/SRPMS/python-${REPO}-${VER}-${TIME}_git${SHA}.src.rpm --resultdir=build-rpm/RPMS $EXTRA
fi

cd $CUR
