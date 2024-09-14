curl -X POST \
     -H "Authorization: Bearer $(gcloud auth application-default print-access-token)" \
     -H "Content-Type: application/json; charset=utf-8" \
     --data "{
  'config': {...},
  'output_config': {
     'gcs_uri':'gs://bucket/result-output-path.json'
  },
  'audio': {
    'uri': 'gs://bucket/audio-path'
  }
}" "https://speech.googleapis.com/v1p1beta1/speech:longrunningrecognize"
