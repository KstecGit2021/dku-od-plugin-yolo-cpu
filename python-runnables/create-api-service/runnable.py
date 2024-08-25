# This file is the actual code for the Python runnable create-api-service
import dataiku
from dataiku.runnables import Runnable
import os
import sys
import shutil
import logging
from api_designer_utils import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,  # avoid getting log from 3rd party module
                    format='object-detection-macro %(levelname)s - %(message)s')


class MyRunnable(Runnable):
    """The base interface for a Python runnable"""

    def __init__(self, project_key, config, plugin_config):
        """
        :param project_key: the project in which the runnable executes
        :param config: the dict of the configuration of the object
        :param plugin_config: contains the plugin settings
        :param client: DSS client
        :param project: DSS project in which the macro is executed
        :param plugin_id: name of the plugin in use
        """
        self.project_key = project_key
        self.config = config
        self.plugin_config = plugin_config
        self.client = dataiku.api_client()
        self.project = self.client.get_project(self.project_key)
        self.plugin_id = "object-detection-cpu-yolo"

    def get_progress_target(self):
        """
        If the runnable will return some progress info, have this function return a tuple of 
        (target, unit) where unit is one of: SIZE, FILES, RECORDS, NONE
        """
        return None

    def run(self, progress_callback):
        """
        Do stuff here. Can return a string or raise an exception.
        The progress_callback is a function expecting 1 value: current progress
        """

        params = get_params(self.config, self.client, self.project)
        root_path = dataiku.get_custom_variables(project_key=self.project_key)['dip.home']

        copy_plugin_to_dss_folder(self.plugin_id, root_path, params.get(
            "model_folder_id"), self.project_key, force_copy=True)
        create_api_code_env(self.plugin_id, root_path, self.client, params.get(
            'code_env_name'), params.get('use_gpu'))
        api_service = get_api_service(params, self.project)
        endpoint_settings = get_model_endpoint_settings(params)
        create_python_endpoint(api_service, endpoint_settings)
        html_str = get_html_result(params)

        return html_str
