export function buildMetadataPatch(form) {
  const payload = { report_type: form.report_type, broker: form.broker }
  if (form.report_type === '个股') {
    payload.security_name = form.security_name
  } else if (form.report_type === '行业') {
    payload.industry_l1 = form.industry_l1
    payload.industry_l2 = form.industry_l2
  }
  return payload
}
