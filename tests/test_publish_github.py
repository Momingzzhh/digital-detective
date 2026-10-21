import pytest
from scripts.publish_github import allowed_public_file, identity


@pytest.mark.parametrize('path', ['.env','.env.cloud.disabled','data/review/answer.json','models/weights.safetensors','runtime/log'])
def test_publication_refuses_private_artifacts(path):
    with pytest.raises(ValueError): allowed_public_file(path,b'private')


def test_publication_keeps_utc_commit_metadata_and_rejects_credentials():
    assert identity('Researcher <1+researcher@users.noreply.github.com> 0 +0000')['date']=='1970-01-01T00:00:00+00:00'
    allowed_public_file('.env.example',b'LOCAL_ADAPTER_PATH=')
    with pytest.raises(ValueError): allowed_public_file('accident.txt',b'sk-'+b'x'*32)
