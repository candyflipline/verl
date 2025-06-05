import json
import logging
import os
import time

import requests
from omegaconf import DictConfig
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from recipe.atropos.exceptions import (AtroposAPIException,
                                       AtroposNoDataFetchException)
from recipe.atropos.schemas import (AtroposConfig, AtroposRegisterPayload,
                                    AtroposRegisterResponse,
                                    AtroposScoredDataBatch)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AtroposClient:
    """
    AtroposClient is a client for the Atropos API.
    """

    HEADERS = {"Content-Type": "application/json"}
    REGISTER_ENDPOINT = "/register"
    FETCH_BATCH_ENDPOINT = "/batch"
    TIMEOUT = 30
    POLLING_INTERVAL_SECONDS = 5
    MAX_BACKOFF_ATTEMPTS = 3
    PROGRESSIVE_BACKOFF_FACTOR = 2

    def __init__(self, verl_config: DictConfig, tokenizer: PreTrainedTokenizerBase):
        assert verl_config is not None, "Verl config is None"
        assert tokenizer is not None, "Tokenizer is None"
        assert hasattr(verl_config, "atropos"), "Atropos config is missing"

        self.atropos_cfg = AtroposConfig.model_validate(verl_config.atropos)

        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)

    def register_trainer_with_atropos(self) -> AtroposRegisterResponse:
        """
        Register the trainer with the Atropos API.
        """
        registration_payload = AtroposRegisterPayload.from_atropos_config(self.atropos_cfg)

        try:
            response = self.session.post(
                f"{self.atropos_cfg.atropos_host}{self.REGISTER_ENDPOINT}",
                json=registration_payload.model_dump(),
                timeout=self.TIMEOUT,
            )
            response.raise_for_status()
            logger.info(f"Successfully registered trainer with Atropos API. Response: {response.json()}")
            response_json = response.json()
            return AtroposRegisterResponse.model_validate(response_json)
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error when registering RL trainer with Atropos API: {e}")
            raise e from e
        except Exception as e:
            logger.error(f"Unexpected error when registering RL trainer with Atropos API: {e}")
            raise e from e

    def fetch_batch(self) -> AtroposScoredDataBatch:
        """
        Fetch a batch of data from Trajectory API. Uses progressive backoff to handle transient errors.
        """
        attempts = 0
        backoff_factor = 1

        while attempts < self.MAX_BACKOFF_ATTEMPTS:
            attempt_str = f"Attempt {attempts + 1}/{self.MAX_BACKOFF_ATTEMPTS}"
            logger.info(f"{attempt_str}: Fetching batch from Atropos API")
            try:
                response = self.session.get(
                    f"{self.atropos_cfg.atropos_host}{self.FETCH_BATCH_ENDPOINT}",
                    timeout=self.TIMEOUT,
                )
                response.raise_for_status()
                data = response.json()
                if data.get("batch") is not None:
                    logger.info(f"{attempt_str}: Batch received from Atropos API")
                    return AtroposScoredDataBatch.model_validate(data)
                else:
                    wait_time = backoff_factor * self.POLLING_INTERVAL_SECONDS
                    logger.info(f"{attempt_str}: Empty batch received from Atropos API")
                    logger.info(f"{attempt_str}: Waiting for {wait_time} seconds before retrying")
                    time.sleep(wait_time)
                    backoff_factor *= self.PROGRESSIVE_BACKOFF_FACTOR
                    attempts += 1
                    continue
            except requests.exceptions.RequestException as e:
                logger.warning(f"Failed to fetch batch from Atropos API: {e}")
                time.sleep(backoff_factor * self.POLLING_INTERVAL_SECONDS)
                backoff_factor *= self.PROGRESSIVE_BACKOFF_FACTOR
                attempts += 1
                continue
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON response from Atropos API /batch: {e}")
                raise AtroposAPIException(f"Failed to decode JSON response from Atropos API /batch: {e}") from e
            except Exception as e:
                logger.error(f"Failed to fetch batch from Atropos API: {e}")
                raise AtroposAPIException(f"Failed to fetch batch from Atropos API: {e}") from e

        logger.error("Failed to fetch batch from Atropos API after all attempts.")
        raise AtroposNoDataFetchException(f"Failed to fetch batch from Atropos API after {self.MAX_BACKOFF_ATTEMPTS} attempts.")
                logger.error(f"Failed to fetch batch from Atropos API: {e}")
                raise AtroposAPIException(f"Failed to fetch batch from Atropos API: {e}") from e

        logger.error("Failed to fetch batch from Atropos API after all attempts.")
        raise AtroposNoDataFetchException(f"Failed to fetch batch from Atropos API after {self.MAX_BACKOFF_ATTEMPTS} attempts.")
