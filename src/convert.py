# https://github.com/kabrabharat/Electric-Pole-detection-using-darknet/tree/master

import os
import numpy as np
import cv2
import supervisely as sly
from supervisely.io.fs import (
    get_file_name_with_ext,
    get_file_name,
    get_file_ext,
    file_exists,
    dir_exists,
    mkdir,
    remove_dir,
)
import xml.etree.ElementTree as ET
from dotenv import load_dotenv

# Path to the original dataset

import supervisely as sly
import gdown
import os
from dataset_tools.convert import unpack_if_archive
import src.settings as s


def download_dataset():
    archive_path = os.path.join(sly.app.get_data_dir(), "archive.zip")

    if not os.path.exists(archive_path):
        if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
            gdown.download(s.DOWNLOAD_ORIGINAL_URL, archive_path, quiet=False)
        if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
            for name, url in s.DOWNLOAD_ORIGINAL_URL:
                gdown.download(url, os.path.join(archive_path, name), quiet=False)
    else:
        sly.logger.info(f"Path '{archive_path}' already exists.")
    return unpack_if_archive(archive_path)


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    # project_name = "electric pole detection"
    dataset_path = (
        "/home/grokhi/rawdata/electric-pole/Electric-Pole-detection-using-darknet/dataset"
    )
    batch_size = 30
    ds_name = "ds"
    images_ext = ".jpg"

    def create_ann(image_path):
        labels = []

        image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = image_np.shape[0]
        img_wight = image_np.shape[1]

        bbox_path = image_path.split(".")[0] + ".txt"

        if file_exists(bbox_path):
            with open(bbox_path) as f:
                content = f.read().split("\n")

                for curr_data in content:
                    if len(curr_data) != 0:
                        curr_data = curr_data.strip()
                        curr_data = list(map(float, curr_data.split(" ")))
                        left = int((curr_data[1] - curr_data[3] / 2) * img_wight)
                        right = int((curr_data[1] + curr_data[3] / 2) * img_wight)
                        top = int((curr_data[2] - curr_data[4] / 2) * img_height)
                        bottom = int((curr_data[2] + curr_data[4] / 2) * img_height)
                        rectangle = sly.Rectangle(top=top, left=left, bottom=bottom, right=right)
                        label = sly.Label(rectangle, obj_class)
                        labels.append(label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels)

    obj_class = sly.ObjClass("electric pole", sly.Rectangle)
    # tag_meta_train = sly.TagMeta("train", sly.TagValueType.NONE)
    # tag_meta_test = sly.TagMeta("test", sly.TagValueType.NONE)

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(obj_classes=[obj_class])
    api.project.update_meta(project.id, meta.to_json())

    for ds_name in ["train", "test"]:
        if ds_name == "test":
            images_names = [
                im_name
                for im_name in os.listdir(dataset_path)
                if (get_file_ext(im_name) == images_ext)
                and (
                    get_file_name(im_name) in ["57", "90", "65", "36", "89", "12", "44", "29", "84"]
                )
            ]
        else:
            images_names = [
                im_name
                for im_name in os.listdir(dataset_path)
                if (get_file_ext(im_name) == images_ext)
                and (
                    get_file_name(im_name)
                    not in ["57", "90", "65", "36", "89", "12", "44", "29", "84"]
                )
            ]
        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

        for images_names_batch in sly.batched(images_names, batch_size=batch_size):
            img_pathes_batch = [
                os.path.join(dataset_path, image_name) for image_name in images_names_batch
            ]

            img_infos = api.image.upload_paths(dataset.id, images_names_batch, img_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            anns = [create_ann(image_path) for image_path in img_pathes_batch]
            api.annotation.upload_anns(img_ids, anns)

            progress.iters_done_report(len(images_names_batch))
    return project
