// 元数据修正提交核心字段：版本、报告类型、发布机构与生效日期
export function buildMetadataPatch(form) {
  return { version: form.version, report_type: form.report_type, broker: form.broker, date: form.date || '' }
}
