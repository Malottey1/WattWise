import os

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "super-secret-dev-key") 
    SQLALCHEMY_DATABASE_URI = (
        "mysql+pymysql://root@localhost/wattwise?unix_socket=/tmp/mysql.sock"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class DevelopmentConfig(Config):
    DEBUG = True


class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"


class ProductionConfig(Config):
    DEBUG = False
