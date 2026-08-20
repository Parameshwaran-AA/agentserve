{{/*
Fail fast on the one misconfiguration that breaks the product silently.
*/}}
{{- define "agentserve.validate" -}}
{{- if and (gt (int .Values.gateway.replicaCount) 1) (not .Values.redis.enabled) -}}
{{- fail "gateway.replicaCount > 1 requires redis.enabled=true. Without a shared session store each gateway pod keeps its own session->replica map, so affinity degrades to round-robin and the cache hit rate collapses without any error surfacing." -}}
{{- end -}}
{{- if and .Values.redis.enabled (not .Values.redis.url) -}}
{{- fail "redis.enabled=true requires redis.url (e.g. redis://agentserve-redis:6379/0)" -}}
{{- end -}}
{{- end -}}

{{- define "agentserve.vllmEndpoints" -}}
{{- $out := list -}}
{{- range $i := until (int .Values.vllm.replicaCount) -}}
{{- $out = append $out (printf "http://%s-vllm-%d.%s-vllm:8000" $.Release.Name $i $.Release.Name) -}}
{{- end -}}
{{- join "," $out -}}
{{- end -}}
