"""Модуль для создания базового класса будущих ембеддинг векторов"""

from pathlib import Path
from typing import Literal

from huggingface_hub import repo_exists, snapshot_download
from sentence_transformers import SentenceTransformer
from src.tools.self_logger import setup_logger

from configs.const import MODEL_CONFIG_TO_EXISTS_CHECK, MODULES_JSON_TO_EXISTS_CHECK
from configs.custom_error import HFValidateTokenError, ModelExistsOnHFError
from core.embeddings_models.model_config import BaseVectorModel

logger = setup_logger(__name__)


class EchoMindEmbedding:
    """Базовый класс для создания любой модели эмбеддингов"""

    def __init__(
        self,
        validate_data: BaseVectorModel,
        quant: bool,
        type_of_quant_backend: Literal["torch", "onnx", "openvino"] = "openvino",
    ) -> None:
        """
        Инициализация данных для создания модели

        1.  Принимаем имя с HF и путь до папки.
        1.1 Проверяем приватность.
        1.2 После успешной проверки приватности:
            1.3. Если модель приватная и есть валидный токен, то переходим в 1.5.
            1.4.    Если нет, то raises
        1.5.    Валидируем полученные значения
        2.  Проверяем есть ли эта модель в этой папке
        3.0  Если есть, то:
            3.1  Создаем экземпляр ST, вместо названия - путь до модели
        3.2 Если нет, то:
            3.3 Скачиваем в папку по адресу через `snapshot_download`.
            3.4 Выполняем пункт `3.1`

        Args:
            validate_data (BaseVectorModel): провалидированные данные, необходимые для создания
            quant (bool): указание о необходимости квантования модели
            type_of_quant_backend(str): встроенный в SentenceTransformer способ квантования

        """
        self._validate_data: BaseVectorModel = validate_data

        self._batch_size: int = validate_data.batch_size
        self._show_progress_bar: bool = validate_data.show_progress_bar
        self.is_model_downloaded: bool = False

        self._model: SentenceTransformer | None = self._create_model()
        self.quant = quant
        self.type_of_quant_backend = type_of_quant_backend

    def _create_model(self) -> SentenceTransformer | None:
        try:
            self.is_model_downloaded = self._validate_exists_local_model()
            if not self.is_model_downloaded and self._check_existing_model_repo():
                self.download_model_for_local_path()
                self.is_model_downloaded = True

            return SentenceTransformer(
                model_name_or_path=self._validate_data.local_path_for_downloads,
                local_files_only=True,
                trust_remote_code=self._validate_data.trust_remote_code,
                device=self._validate_data.device,
            )

        except HFValidateTokenError as ve:
            msg = f"Ошибка при скачивании приватной модели {ve}"
            logger.exception(msg)
            return None

        except ModelExistsOnHFError as ee:
            msg = f"Ошибка при скачивании модели, модель не существует {ee}"
            logger.exception(msg)
            return None

        except ValueError as ve:
            msg = f"Ошибка значения {ve}"
            logger.exception(msg)
            return None

    def _check_existing_model_repo(self) -> bool:
        """
        Проверяем есть лит акая модель в HF hub.

        model_name (str): Название модели разделенное `/`.
        token (str):
                Токен для доступа к приватной модели HF
                (https://huggingface.co/docs/huggingface_hub/quick-start#authentication).
                Для неприватной модели используем None

        Returns:
            True, если модель существует

        """
        model_name = self._validate_data.models_name_from_hf
        token = self._validate_data.hf_token

        result = repo_exists(
            model_name,
            token=token,
        )
        if not result:
            msg = (
                f"Модель: {model_name} не существует в HF.\n"
                f"Убедитесь, что вы ввели корректное название"
            )
            logger.error(msg)

        return result

    def _validate_exists_local_model(self) -> bool:
        """
        Проверяем есть ли модель в указанной локальной папке

        local_path (str | Path): Путь, где хранится / будет храниться модель эмбеддинга
        model_name (str): Название модели для красивых логов

        Returns:
            bool: Если True, значит модель уже есть. Если False, то модель отсутствует.

        """
        local_path: Path = self._generate_local_path()
        model_dir = Path(local_path)
        required_files = [
            model_dir / MODULES_JSON_TO_EXISTS_CHECK,
            model_dir / MODEL_CONFIG_TO_EXISTS_CHECK,
        ]
        result = all(file.exists() for file in required_files)
        if not result:
            msg = f"Отсутствует скачанная модель {self._validate_data.models_name_from_hf} по пути: {local_path}"
            logger.warning(msg)

        return result

    def _generate_local_path(self) -> Path:
        """
        Функция помощник, которая генерирует правильное название структуры хранения моделей.
        Структура: папка заданная пользователем (или папка по-умолчанию) -> название модели ->
        спец папка для квантованных моделей

        Returns:
            Path: путь конечной папки выбранной модели

        """
        # папка / название модели / quant (если да)
        model_name = self._validate_data.models_name_from_hf
        path = self._validate_data.local_path_for_downloads / model_name
        return (path / "quant") if self.quant else path

    def download_model_for_local_path(self) -> None:
        """
        Функция сохранения модели в локальную папку

        token (Optional[str]): токен для доступа к приватной модели HF
        model_name_from_hf (str): название модели с сайта HF
        path_to_download_model (str|Path): Путь или строка пути до папки с местом хранения моделей

        Returns:
            None

        """
        model_name_from_hf = self._validate_data.models_name_from_hf
        path_to_download_model = self._generate_local_path()

        if not self._check_existing_model_repo():
            msg = (
                f"Модель: {model_name_from_hf} не существует в HF.\n"
                'Убедитесь, что название модели имеет форму "владелец/модель", '
                'или "компания/модель"'
            )
            raise ModelExistsOnHFError(msg)

        if self._validate_exists_local_model():
            msg = f"Модель {model_name_from_hf} уже существует локально"
            logger.info(msg)

        else:
            msg = f"Начинаю загрузку модели {model_name_from_hf}"
            logger.info(msg)

            local_path = path_to_download_model
            parent_path = local_path.parent if self.quant else local_path

            snapshot_download(
                repo_id=model_name_from_hf,
                local_dir=parent_path,
                cache_dir=parent_path,
            )
            if self.quant:
                quant_model = SentenceTransformer(
                    model_name_or_path=str(parent_path),
                    backend=self.type_of_quant_backend,
                )
                msg = f"Модель {model_name_from_hf} квантована"
                logger.info(msg)
                quant_model.save_pretrained(str(local_path))

            msg = (
                f"Модель {model_name_from_hf} загружена по пути: {local_path if self.quant else parent_path}.\n"
                f"Было применено квантование -> {'Да' if self.quant else 'Нет'}."
            )
            logger.info(msg)


if __name__ == "__main__":
    frida_folder: str = "frida"
    model_name_frida = "ai-forever/FRIDA"
