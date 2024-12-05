import openi
from openmind_hub import snapshot_download


def modelers2openi(model_id, openi_repo_id="starmountain1997/AMP"):
    _, model_name = model_id.split("/")
    path = snapshot_download(model_id)
    openi.upload_model(openi_repo_id, model_name, path)


if __name__ == "__main__":
    modelers2openi("openMind-ecosystem/Yi-6B")
    modelers2openi("openMind-ecosystem/Yi-1.5-9b-chat")
    modelers2openi("openMind-ecosystem/codegeex4-all-9b")
