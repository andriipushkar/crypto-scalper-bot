{{/*
Expand the name of the chart.
*/}}
{{- define "crypto-scalper-bot.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "crypto-scalper-bot.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "crypto-scalper-bot.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "crypto-scalper-bot.labels" -}}
helm.sh/chart: {{ include "crypto-scalper-bot.chart" . }}
{{ include "crypto-scalper-bot.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "crypto-scalper-bot.selectorLabels" -}}
app.kubernetes.io/name: {{ include "crypto-scalper-bot.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Trading bot labels
*/}}
{{- define "crypto-scalper-bot.tradingBot.labels" -}}
{{ include "crypto-scalper-bot.labels" . }}
app.kubernetes.io/component: trading-bot
{{- end }}

{{- define "crypto-scalper-bot.tradingBot.selectorLabels" -}}
{{ include "crypto-scalper-bot.selectorLabels" . }}
app.kubernetes.io/component: trading-bot
{{- end }}

{{/*
Dashboard labels
*/}}
{{- define "crypto-scalper-bot.dashboard.labels" -}}
{{ include "crypto-scalper-bot.labels" . }}
app.kubernetes.io/component: dashboard
{{- end }}

{{- define "crypto-scalper-bot.dashboard.selectorLabels" -}}
{{ include "crypto-scalper-bot.selectorLabels" . }}
app.kubernetes.io/component: dashboard
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "crypto-scalper-bot.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "crypto-scalper-bot.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Redis host
*/}}
{{- define "crypto-scalper-bot.redisHost" -}}
{{- if .Values.redis.enabled }}
{{- printf "%s-redis-master" (include "crypto-scalper-bot.fullname" .) }}
{{- else }}
{{- .Values.externalRedis.host }}
{{- end }}
{{- end }}

{{/*
PostgreSQL host
*/}}
{{- define "crypto-scalper-bot.postgresqlHost" -}}
{{- if .Values.postgresql.enabled }}
{{- printf "%s-postgresql" (include "crypto-scalper-bot.fullname" .) }}
{{- else }}
{{- .Values.externalPostgresql.host }}
{{- end }}
{{- end }}

{{/*
Image pull secrets
*/}}
{{- define "crypto-scalper-bot.imagePullSecrets" -}}
{{- if .Values.global.imagePullSecrets }}
imagePullSecrets:
{{- range .Values.global.imagePullSecrets }}
  - name: {{ . }}
{{- end }}
{{- end }}
{{- end }}
