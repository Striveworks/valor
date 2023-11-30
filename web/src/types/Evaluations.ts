export type Evaluation = {
    dataset: string
    model: string
}

export type EvaluationSettings = {
    taskType: string
    parameters: EvaluationSettingsParams
    filters: any
    jobId: number
    status: string
    metrics: any
    confusionMatrices: any

}

export type EvaluationSettingsParams = {
    iou_thresholds_to_compute : number[]
    iou_thresholds_to_keep: number[]
}


