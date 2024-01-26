import logging
import multiprocessing
import os
import time
import warnings
from multiprocessing import freeze_support

from dotenv import load_dotenv
from minio import Minio

from src.common.minio_fncts.minio_api_calls import upload_object
from src.common.persisted_memory import PersistedMemory, SESSION_DIR_NAME, SESSION_NAME_PREFIX, TIMESTEP_NAME_PREFIX, \
    GENERATION_NAME_PREFIX, AGENT_NAME_PREFIX, IMAGE_ENDING, TRANSITION_ENDING

warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_ADDRESS = os.getenv("MINIO_ADDRESS")
MINIO_PORT = os.getenv("MINIO_PORT")


def get_minio_client_secure_no_cert():
    logging.info("\nS3 connect:")
    logging.info("address: " + MINIO_ADDRESS)
    logging.info("port: " + MINIO_PORT)
    minio_client: Minio = Minio(
        MINIO_ADDRESS + ":" + MINIO_PORT,
        secure=True,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        cert_check=False
    )
    return minio_client


def minio_check_bucket(minio_client: Minio, bucket_name: str):
    bucket_exists = False
    logging.info("\nS3 available buckets")
    for bucket in minio_client.list_buckets():
        logging.info(str(bucket.name) + " " + str(bucket.creation_date))

        if bucket.name == bucket_name:
            logging.info("\nS3 connection established:")
            logging.info("bucket: " + bucket_name)
            bucket_exists = True
    if not bucket_exists:
        raise Exception("S3 bucket found: " + bucket_name)


def minio_bucket_get_all_paths(minio_client: Minio, bucket_name: str, prefix=""):
    # get all already existing paths using one request (instead of checking each single path with a request)
    logging.info("Getting existing paths from S3 bucket " + bucket_name)
    s3_existing_transition_paths = []
    s3_existing_json_count = 0
    s3_existing_png_count = 0
    minio_objects = minio_client.list_objects(bucket_name, recursive=True, prefix=prefix)
    for item in minio_objects:
        path: str = item.object_name
        s3_existing_transition_paths.append(path)
        if path.endswith(".json"):
            s3_existing_json_count += 1
        if path.endswith(".png"):
            s3_existing_png_count += 1
    logging.info("Found " + str(len(s3_existing_transition_paths)) + " paths in S3 bucket " + bucket_name)
    logging.info("Found " + str(s3_existing_json_count) + " json files in S3 bucket " + bucket_name)
    logging.info("Found " + str(s3_existing_png_count) + " png files in S3 bucket " + bucket_name)

    return s3_existing_transition_paths


def abs_path_to_bucket_rel_path(file_abs_path, local_filesystem_store_root_dir):
    rel_bucket_file_path = file_abs_path.replace(local_filesystem_store_root_dir, "")
    rel_bucket_file_path = rel_bucket_file_path.replace('\\', '/')
    return rel_bucket_file_path


def upload_transitions_to_minio(minio_client: Minio, bucket_name: str, local_filesystem_store_root_dir: str,
                                min_transitions=0, validate_paths=True):
    """
    @param minio_client: minio client
    @param bucket_name: minio bucket name
    @param local_filesystem_store_root_dir: abs path to lokal directory containing hierarchical saved transitions
    @param min_transitions: Folder should have this minimum number of transitions to be uploaded
    """
    minio_check_bucket(minio_client, bucket_name)
    s3_existing_transition_paths = minio_bucket_get_all_paths(minio_client, bucket_name)
    transition_abs_paths_grouped, session_id, generation_id, agent_id = PersistedMemory.get_timestep_paths_from_filesystem_by_filter(
        local_filesystem_store_root_dir=local_filesystem_store_root_dir, session_id=None, generation_id=None,
        agent_id=None,
        group_paths=True)
    """
    STEP 1: Filter out invalid paths
    """
    logging.info("Filtering upload file paths")

    filtered_transition_abs_paths_grouped = []
    total_on_device = 0
    already_existing_count = 0
    for agent_transition_paths in transition_abs_paths_grouped:
        total_on_device += len(agent_transition_paths)

    if validate_paths:
        for transition_abs_paths in transition_abs_paths_grouped:
            if len(transition_abs_paths) < min_transitions:
                logging.warning(
                    "Not enough transitions: Skipped agent with " + str(len(transition_abs_paths)) + " transitions")
                if len(transition_abs_paths) > 0:
                    logging.warning(transition_abs_paths[0])
            else:
                transition_abs_paths = [p for p in transition_abs_paths if PersistedMemory.is_transition_valid(p)]
                filtered_transition_abs_paths = []
                for path in transition_abs_paths:
                    transition_file, view_file = PersistedMemory.get_transition_file_paths(path)

                    s3_transition_file = abs_path_to_bucket_rel_path(transition_file, local_filesystem_store_root_dir)
                    s3_view_file = abs_path_to_bucket_rel_path(view_file, local_filesystem_store_root_dir)
                    json_found = s3_transition_file in s3_existing_transition_paths
                    img_found = s3_view_file in s3_existing_transition_paths
                    if not json_found or not img_found:
                        filtered_transition_abs_paths.append(path)
                    else:
                        already_existing_count += 1
                filtered_transition_abs_paths_grouped.append(filtered_transition_abs_paths)
    else:
        filtered_transition_abs_paths_grouped = transition_abs_paths_grouped

    """
    STEP 2: Upload
    """
    logging.info(str(already_existing_count) + "/" + str(
        total_on_device) + " transitions are already existing in S3 bucket " + bucket_name)
    total_on_device_for_upload = 0
    for agent_transition_paths in filtered_transition_abs_paths_grouped:
        total_on_device_for_upload += len(agent_transition_paths)
    logging.info("Uploading " + str(total_on_device_for_upload) + " transitions")
    count_done = 0
    n = 50
    if total_on_device_for_upload < n:
        n = 1
    for agent_transition_paths in filtered_transition_abs_paths_grouped:
        for agent_transition_path in agent_transition_paths:
            start_time = time.time()
            transition_file, view_file = PersistedMemory.get_transition_file_paths(agent_transition_path)

            logs = False
            if count_done % n == 0 and count_done > n:
                logs = True
            already_exists = False
            try:
                s3_transition_file = abs_path_to_bucket_rel_path(transition_file, local_filesystem_store_root_dir)
                s3_view_file = abs_path_to_bucket_rel_path(view_file, local_filesystem_store_root_dir)
                if os.path.isfile(transition_file) and os.path.isfile(view_file):
                    already_exists = upload_object(minio_client=minio_client, bucket_name=bucket_name,
                                                   object_name=s3_transition_file,
                                                   file_path=transition_file, overwrite=False, logs=logs)

                    already_exists = upload_object(minio_client=minio_client, bucket_name=bucket_name,
                                                   object_name=s3_view_file,
                                                   file_path=view_file, overwrite=False, logs=logs)
                else:
                    logging.warning("Files not existing: " + transition_file + "   " + view_file)
            except Exception as e:
                print(e)

            if logs:
                logging.info("Try upload transition " + str(count_done) + "/" + str(total_on_device_for_upload)
                             + " needed for 1 upload: " + str(time.time() - start_time) + "s already exists:" + str(
                    already_exists))
            count_done += 1
    logging.info("Upload finished")


def recurse_file_tree(res, array):
    """
    To build a hierarchy of files
    @param res: Result
    @param array: Array containing path parts
    @return:
    """
    if len(array) == 0:
        return
    elif len(array) == 1:
        res.append(array[0])  # leaf of the tree is an array
    else:
        recurse_file_tree(res.setdefault(array[0], [] if len(array) == 2 else {}), array[1:])  # next branch


class Consumer(multiprocessing.Process):

    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.TIMEOUT = 120

    def check_timeout(self, start):
        not_over = time.time() - start <= self.TIMEOUT

        if not not_over:
            pass
            # print(self.name + " timeout")

        return not_over

    def run(self):
        start = time.time()
        while self.check_timeout(start):
            next_task = self.task_queue.get()
            # print(self.name + " started")
            try:
                if "fnc" in next_task:
                    # print(next_task["fnc"])
                    answer = self.calculate(next_task)
                    self.result_queue.put(answer)
                    # print(str(self.name) + " process exiting")
                    self.task_queue.task_done()
                    break
            except Exception as e:
                # print(e)
                # logging.error(e)
                break

        # print(str(self.name) + " done")
        return

    def calculate(self, function_dict: dict):
        function = function_dict["fnc"]
        params = function_dict["params"]
        return function(**params)


def get_s3_transition_rel_paths_grouped_by_agent(minio_client: Minio, bucket_name: str):
    """
    Returns the relative transition paths grouped by agent which are stored in the minio bucket.
    Validation: Each transition path is only returned if files with the following extensions exist:
    IMAGE_ENDING, TRANSITION_ENDING (e.g. .png and .json)
    @param minio_client: minio client
    @param bucket_name: minio bucket name
    @return:
    """
    freeze_support()
    # Establish communication queues
    tasks = multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()



    logging.info("get_s3_transition_rel_paths_grouped_by_agent ...")
    sessions_paths_prefix = SESSION_DIR_NAME + "/" + SESSION_NAME_PREFIX
    files: [] = minio_bucket_get_all_paths(minio_client, bucket_name, prefix=sessions_paths_prefix)

    file_tree: dict = {}
    logging.info("creating file tree")
    for f in files:
        recurse_file_tree(file_tree, f.split("/"))

    sessions_dict: dict = file_tree["sessions"]

    # get transition paths grouped by agent relative to S3 bucket
    s3_transition_rel_paths_grouped_by_agent = []
    session_counter: int = 0
    num_jobs = 0
    todo_list = []

    for session, generations_dict in sessions_dict.items():
        session_counter += 1
        session_id = session.replace(SESSION_NAME_PREFIX, "")
        logging.info("session " + str(session_counter) + "/" + str(len(sessions_dict.items())))
        for generation, agents_dict in generations_dict.items():
            generation_id = generation.replace(GENERATION_NAME_PREFIX, "")

            item = {"agents_dict": agents_dict,
                    "generation_id": generation_id,
                    "session_id": session_id
                    }
            todo_list.append(item)

    # assign tasks to cpu threads until al tasks are done
    while len(todo_list) > 0:

        # Start new consumers
        available_cpus = multiprocessing.cpu_count() - 1  # -1 for main thread
        num_consumers = min(available_cpus, 48)
        consumers = [Consumer(tasks, results) for i in range(num_consumers)]
        for w in consumers:
            w.start()

        while num_consumers > num_jobs and len(todo_list) > 0:
            item = todo_list.pop(0)
            print("task to do: " + str(len(todo_list)))
            num_jobs += 1
            print("num_jobs: " + str(num_jobs))
            tasks.put(
                {"fnc": process_generation_paths,
                 "params":
                     item
                 }
            )

        start = time.time()
        TIMEOUT = 12
        while num_jobs > 0:
            try:
                result = results.get(block=False, timeout=1)
                s3_transition_rel_paths_grouped_by_agent.append(result)
                num_jobs -= 1
                diff = time.time() - start
                print("remaining time: " + str(diff) + " num_jobs: " + str(num_jobs))
                if diff > TIMEOUT:
                    break
                if num_jobs <= 0:
                    break
            except Exception as e:
                # print(e)
                pass
        print("time needed: " + str(time.time() - start))
        print("task to do: " + str(len(todo_list)))

        # Stop consumers
        for w in consumers:
            w.terminate()
            num_consumers -= 1
            print("remaining consumer processes: " + str(num_consumers))
        print("processes finished")

    print("s3_transition_rel_paths_grouped_by_agent: " + str(len(s3_transition_rel_paths_grouped_by_agent)))

    # Wait for all of the tasks to finish
    # tasks.join()
    print("tasks joined")

    return s3_transition_rel_paths_grouped_by_agent


def process_generation_paths(agents_dict, generation_id, session_id):
    print("processing " + str(generation_id))
    for agent, files_list in agents_dict.items():
        agent_id = agent.replace(AGENT_NAME_PREFIX, "")
        agent_dir: str = PersistedMemory.get_agent_rel_path(session_id, generation_id, agent_id)
        logging.info("get_s3_transition_rel_paths_grouped_by_agent: " + agent_dir)
        agent_timestep_files = []
        for f in files_list:
            agent_timestep_files.append(agent_dir + "/" + f)
        agent_timestep_paths = []

        files_total = len(agent_timestep_files)
        counter = 0
        for f in agent_timestep_files:
            counter += 1
            if counter % 1000 == 0:
                print(
                    str(session_id) + " " + str(generation_id) + " " + str(agent_id) + ": " + str(counter) + "/" + str(
                        files_total))
            path = os.path.splitext(f)[0]
            if path not in agent_timestep_paths:
                # Validation if all files for one transition are available
                if path.split("/")[-1] + IMAGE_ENDING in files_list and path.split("/")[
                    -1] + TRANSITION_ENDING in files_list:
                    agent_timestep_paths.append(path)
                else:
                    print("Ignored invalid transition: " + str(path))

        print(str(session_id) + " " + str(generation_id) + " " + str(agent_id) + ": " + str(counter) + "/" + str(
                files_total))
        agent_timestep_paths = sorted(agent_timestep_paths,
                                      key=lambda i: int(i.split(TIMESTEP_NAME_PREFIX)[-1]))

        return agent_timestep_paths
        # s3_transition_rel_paths_grouped_by_agent.append(agent_timestep_paths)
