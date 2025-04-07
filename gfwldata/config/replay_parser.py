from pydantic import BaseModel, Field


class ReplayParserSettings(BaseModel):
    # Replay parser settings
    LABEL_ENCODER_PATH: str = Field()
    VECTORIZER_PATH: str = Field()
    MODEL_PATH: str = Field()

    # Multiprocessing
    MP_PROCESSES: int = Field()


replay_parser_settings = ReplayParserSettings(
    LABEL_ENCODER_PATH="gfwldata/models/deck_classifier/label_encoder.joblib",
    VECTORIZER_PATH="gfwldata/models/deck_classifier/tfidf.joblib",
    MODEL_PATH="gfwldata/models/deck_classifier/xgboost_model.joblib",
    MP_PROCESSES=18,
)
