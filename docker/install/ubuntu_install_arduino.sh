#!/bin/bash
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

set -e
set -u
set -o pipefail

export DEBIAN_FRONTEND=noninteractive
apt-get install -y ca-certificates

# Install arduino-cli latest version
wget -O - https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh -s

# Board manager URLs, separated by semicolons
BOARD_MANAGER_URLS=https://github.com/sonydevworld/spresense-arduino-compatible/releases/download/generic/package_spresense_index.json

# Install supported cores from those URLS
arduino-cli core install SPRESENSE:spresense --additional-urls ${BOARD_MANAGER_URLS}
arduino-cli core install arduino:mbed_nano --additional-urls ${BOARD_MANAGER_URLS}
