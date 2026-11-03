#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: .env file not found at '$ENV_FILE'." >&2
    exit 1
fi

source "$ENV_FILE"
TARGET_PATH="${KITTI_360_DATASET_DIR:?Error: 'KITTI_360_DATASET_DIR' is not set in .env.}"

if [[ ! -d "$TARGET_PATH" ]]; then
    echo "Error: Path '$TARGET_PATH' does not exist." >&2
    exit 1
fi

download_and_extract() {
    local url="$1"
    local archive="$TARGET_PATH/$2"
    local destination="$TARGET_PATH/$3"

    wget -q -O "$archive" "$url"
    mkdir -p "$destination"
    unzip -q "$archive" -d "$destination"
    rm "$archive"
}

download_and_extract \
    "https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/384509ed5413ccc81328cf8c55cc6af078b8c444/calibration.zip" \
    "calibration.zip" "calibration"

download_and_extract \
    "https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/89a6bae3c8a6f789e12de4807fc1e8fdcf182cf4/data_poses.zip" \
    "data_poses.zip" "data_poses"

download_and_extract \
    "https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/6489aabd632d115c4280b978b2dcf72cb0142ad9/data_3d_semantics.zip" \
    "data_3d_semantics.zip" "data_3d_semantics"

download_and_extract \
    "https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/ed180d24c0a144f2f1ac71c2c655a3e986517ed8/data_2d_semantics.zip" \
    "data_2d_semantics.zip" "data_2d_semantics"

velodyne_archive="$TARGET_PATH/download_3d_velodyne.zip"
velodyne_dir="$TARGET_PATH/download_3d_velodyne"
raw_dir="$TARGET_PATH/data_3d_raw"
wget -q -O "$velodyne_archive" \
    "https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/a1d81d9f7fc7195c937f9ad12e2a2c66441ecb4e/download_3d_velodyne.zip"
mkdir -p "$velodyne_dir" "$raw_dir"
unzip -q "$velodyne_archive" -d "$velodyne_dir"
mv "$velodyne_dir"/* "$raw_dir"/
rm "$velodyne_archive"
rmdir "$velodyne_dir"