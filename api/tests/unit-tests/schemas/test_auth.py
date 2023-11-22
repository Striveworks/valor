from velour_api import schemas


# @TODO
def test_auth_User():
    # valid
    schemas.User(email="somestring")
