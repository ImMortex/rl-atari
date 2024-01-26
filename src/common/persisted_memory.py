import copy
import logging
import os
from datetime import date
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np

from src.common.helpers.helpers import load_from_json_file, save_dict_as_json
from src.common.observation_keys import view_key
from src.common.transition import Transition

# constants
SESSION_DIR_NAME = "sessions"
SESSION_NAME_PREFIX = "session_"
GENERATION_NAME_PREFIX = "generation_"
AGENT_NAME_PREFIX = "agent_"
TIMESTEP_NAME_PREFIX = "timestep_"
SESSION_META_DATA_FEATURE_ID = "meta"
TRANSITION_FEATURE_ID = "transition"

IMAGE_ENDING = '.png'
TRANSITION_ENDING = ".json"


class PersistedMemory:

    def __init__(self,
                 img_shape,
                 session_id: str = None,
                 agent_id: str = None,
                 generation_id: str = None,
                 bucket_name: str = "bucket_ma_christian_gurski",
                 local_filesystem_store_root_dir: str = "./tmp/storage/",
                 persist_to_local_filesystem: bool = True):
        """
        @param img_shape: DEPRECATED
        @param session_id: each day is one session
        @param generation_id: parallel running agents with synced start are one generation
        @param agent_id: agent id (e.g. hostname)
        @param bucket_name: DEPRECATED
        @param local_filesystem_store_root_dir: dir on large hard disk to persist data
        @param persist_to_local_filesystem: DEPRECATED
        """

        if session_id is None or session_id == "":
            self.session_id: str = self.get_session_id_today()
        else:
            self.session_id = session_id

        self.bucket_name: str = bucket_name

        self.agent_id = agent_id
        self.generation_id = generation_id

        if self.agent_id is None:
            self.agent_id = "0"
        if self.generation_id is None:
            self.generation_id = "0"

        self.transitions = []  # transitions of the agent in RAM
        self.local_filesystem_store_root_dir = local_filesystem_store_root_dir

    def get_rel_agent_path(self):
        return self.get_agent_rel_path(self.session_id, self.generation_id, self.agent_id)

    @staticmethod
    def get_session_id_today():
        return date.today().strftime('%Y-%m-%d.%H_%M_%S')

    @staticmethod
    def get_agent_timestep_rel_path(rel_agent_path: str, timestep_id: int):
        """
        Returns relative path from bucket to timestep_id of the given agent
        """
        return rel_agent_path + "/" + TIMESTEP_NAME_PREFIX + str(timestep_id)

    @staticmethod
    def get_agent_rel_path(session_id: str, generation_id: str, agent_id: str):
        """
        Returns relative path from bucket to agent_id
        """
        return PersistedMemory.get_session_rel_path(session_id) + "/" \
               + GENERATION_NAME_PREFIX + generation_id + "/" + AGENT_NAME_PREFIX + agent_id

    @staticmethod
    def get_session_meta_data_rel_path(session_id: str):
        return PersistedMemory.get_session_rel_path(session_id) + "/" + SESSION_META_DATA_FEATURE_ID

    @staticmethod
    def get_session_rel_path(session_id):
        return SESSION_DIR_NAME + "/" + SESSION_NAME_PREFIX + session_id

    def timestep_abs_path(self, rel_agent_path, timestep_id):
        return os.path.join(self.local_filesystem_store_root_dir,
                            self.get_agent_timestep_rel_path(rel_agent_path, timestep_id))

    def get_session_meta_data_abs_path(self, session_id):
        return os.path.join(self.local_filesystem_store_root_dir, self.get_session_meta_data_rel_path(session_id))

    @staticmethod
    def sessions_dir_abs_path(local_filesystem_store_root_dir):
        return os.path.join(local_filesystem_store_root_dir, SESSION_DIR_NAME)

    def save_timestep_in_ram(self, transition: Transition):
        self.transitions.append(copy.deepcopy(transition))  # save immutable copy

    def save_from_ram_to_persisted(self, only_delete: bool = False, generation_id=None):
        if generation_id is not None:
            self.generation_id = generation_id

        logging.info("Persisting transitions to " + str(self.get_rel_agent_path()) + " ...")
        if not only_delete:
            self.persist_transitions(self.transitions)

        self.transitions = []  # clear RAM
        return self.transitions

    def persist_transitions(self, transitions: []):
        images: [] = []
        for m in range(len(transitions)):
            if len(transitions) % 100 == 0:
                logging.info("Persisting... " + str(len(transitions)) + " remaining")
            transition: Transition = transitions[m]
            image: np.ndarray = transition.state[view_key]
            images.append(image)
            t_abs_path: str = self.timestep_abs_path(self.get_rel_agent_path(), transition.t)
            self.save_timestep(transition=transition, t_abs_path=t_abs_path)

    @staticmethod
    def save_timestep(transition: Transition, t_abs_path: str):
        transition_dict, pov = PersistedMemory.serialize_transition(transition)
        save_dict_as_json(transition_dict, None, t_abs_path + TRANSITION_ENDING)
        if pov is not None:
            cv2.imwrite(t_abs_path + IMAGE_ENDING, pov)

    @staticmethod
    def serialize_transition(transition: Transition) -> (dict, np.ndarray):
        pov: np.ndarray = None
        if view_key in transition.state:
            pov = transition.state[view_key]
        transition_dict = copy.deepcopy(transition.__dict__)
        transition_dict["state"][view_key] = None
        return transition_dict, pov

    @staticmethod
    def deserialize_transition(transition_dict: dict, image: np.ndarray) -> Transition:
        if image is not None:
            transition_dict["state"][view_key] = image
        return Transition(**transition_dict)

    def load_timestep(self, session_id: str, generation_id: str, agent_id: str, timestep_id: int) -> Transition:
        rel_agent_path = self.get_agent_rel_path(self.session_id, generation_id, agent_id)
        transition = None
        try:
            abs_path: str = self.timestep_abs_path(rel_agent_path, timestep_id)
            transition: Transition = self.get_transition_from_abs_path(abs_path)
        except FileNotFoundError as e:
            logging.error(e)
            return None

        return transition

    @staticmethod
    def get_transition_from_abs_path(abs_path) -> Transition:
        transition_available, transition_file, view_file = PersistedMemory.is_transition_available(abs_path)

        if transition_available:
            transition_dict: dict = load_from_json_file(transition_file)
            pov: np.ndarray = cv2.imread(view_file, 1)
            transition: Transition = PersistedMemory.deserialize_transition(transition_dict, pov)
            return transition

        return None

    @staticmethod
    def is_transition_available(transition_abs_path):
        transition_file, view_file = PersistedMemory.get_transition_file_paths(transition_abs_path)
        transition_available: bool = os.path.isfile(transition_file) and os.path.isfile(transition_file)
        return transition_available, transition_file, view_file

    @staticmethod
    def get_transition_file_paths(transition_path):
        transition_file = transition_path + TRANSITION_ENDING
        view_file = transition_path + IMAGE_ENDING
        return transition_file, view_file

    @staticmethod
    def is_transition_valid(transition_abs_path) -> bool:
        """
        Shortcut for is_transition_available
        """
        transition_available, t_file, i_file = PersistedMemory.is_transition_available(transition_abs_path)
        return transition_available

    @staticmethod
    def get_timestep_paths_from_filesystem_by_filter(local_filesystem_store_root_dir: str, session_id: str,
                                                     generation_id: str = None,
                                                     agent_id: str = None, group_paths: bool = False) -> []:
        """
        Load timestep transitions from file system
        @param group_paths: group paths by agent using additional array dimension
        @param local_filesystem_store_root_dir: data dir
        @param session_id: None or specify session_id as filter
        @param generation_id:  None or specify generation_id as filter
        @param agent_id:  None or specify agent_id as filter
        @return:
        """
        sessions_dir = PersistedMemory.sessions_dir_abs_path(local_filesystem_store_root_dir)
        session_dirs = [f.path for f in os.scandir(sessions_dir) if f.is_dir()]

        # filter
        transition_abs_paths = []
        for session_dir in session_dirs:
            if session_id is None or (session_id is not None and SESSION_NAME_PREFIX + session_id in session_dir):
                generations = [f.path for f in os.scandir(session_dir) if f.is_dir()]

                for generation_dir in generations:
                    if generation_id is None or (
                            generation_id is not None and GENERATION_NAME_PREFIX + generation_id in generation_dir):
                        agents = [f.path for f in os.scandir(generation_dir) if f.is_dir()]

                        for agent_dir in agents:
                            if len(generations) < 20:
                                print("searching", agent_dir)

                            if agent_id is None or (
                                    agent_id is not None and AGENT_NAME_PREFIX + agent_id in agent_dir):
                                agent_timestep_files = [join(agent_dir, f.split(".json")[0]) for f in listdir(agent_dir)
                                                        if isfile(join(agent_dir, f)) and ".json" in f]

                                PersistedMemory.add_abs_transition_paths_for_agent(agent_dir, agent_timestep_files,
                                                                                   transition_abs_paths, group_paths)

        return transition_abs_paths, session_id, generation_id, agent_id

    @staticmethod
    def add_abs_transition_paths_for_agent(agent_dir, agent_timestep_files, transition_abs_paths,
                                           group_paths: bool = False):
        agent_t_abs_paths = []

        if len(agent_timestep_files) >= 2:
            first: Transition = PersistedMemory.get_transition_from_abs_path(agent_timestep_files[0])
            last: Transition = PersistedMemory.get_transition_from_abs_path(agent_timestep_files[-1])

            begin = first.t
            end = max(last.t, len(agent_timestep_files))
            # keep order by timestep numeric id
            for t in range(begin, end):
                abs_path = os.path.join(agent_dir, TIMESTEP_NAME_PREFIX + str(t))
                if not group_paths:
                    transition_abs_paths.append(abs_path)
                else:
                    agent_t_abs_paths.append(abs_path)
        if group_paths:
            transition_abs_paths.append(agent_t_abs_paths)

    @staticmethod
    def get_transition_id_from_path(path):
        splits: [] = path.split(TIMESTEP_NAME_PREFIX)
        return int(splits[-1])

    @staticmethod
    def get_transition_neighbour_path(path, neighbour_id: int) -> str:
        splits: [] = path.split(TIMESTEP_NAME_PREFIX)
        return os.path.join(splits[0], TIMESTEP_NAME_PREFIX + str(neighbour_id))


