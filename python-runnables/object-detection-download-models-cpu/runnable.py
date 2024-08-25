# -*- coding: utf-8 -*-
import sys
import requests
import json
import os
import os.path as op

from dataiku.runnables import Runnable
import dataiku
import pandas as pd

import constants
import download_utils as dl_utils


class MyRunnable(Runnable):
    """The base interface for a Python runnable"""

    def __init__(self, project_key, config, plugin_config):
        """
        :param project_key: the project in which the runnable executes
        :param config: the dict of the configuration of the object
        :param plugin_config: contains the plugin settings
        """
        self.project_key = project_key
        self.config = config
        self.plugin_config = plugin_config
        self.client = dataiku.api_client()


    def get_progress_target(self):
        """
        If the runnable will return some progress info, have this function return a tuple of
        (target, unit) where unit is one of: SIZE, FILES, RECORDS, NONE
        """
        return (100, 'NONE')


    def run(self, progress_callback):
        # Retrieving parameters
        output_folder_name = self.config['folder_name']
        model = self.config['model']

        architecture, trained_on = model.split('_')

        # Creating new Managed Folder if needed
        project = self.client.get_project(self.project_key)

        for folder in project.list_managed_folders():
            if output_folder_name == folder['name']:
                output_folder = project.get_managed_folder(folder['id'])
                break
        else:
            output_folder = project.create_managed_folder(output_folder_name)

        output_folder_path = dataiku.Folder(output_folder.get_definition()["id"], project_key=self.project_key).get_path()

        # Building config file
        config = {
            "architecture": architecture,
            "trained_on": trained_on
        }

        dl_utils.download_labels(trained_on, op.join(output_folder_path, constants.LABELS_FILE))

        # Download weights from s3 (dataiku-labs-public).
        dl_utils.download_model(architecture, trained_on,
                                op.join(output_folder_path, constants.WEIGHTS_FILE),
                                progress_callback)


        output_folder.put_file(constants.CONFIG_FILE, json.dumps(config))

        return "<span>DONE</span>"