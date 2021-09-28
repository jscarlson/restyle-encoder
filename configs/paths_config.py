dataset_paths = {
	'ffhq': '',
	'celeba_test': '',

	'cars_train': '',
	'cars_test': '',

	'church_train': '',
	'church_test': '',

	'horse_train': '',
	'horse_test': '',

	'afhq_wild_train': '',
	'afhq_wild_test': '',

	'font_train': '/mnt/data01/AWS_S3_CONTAINER/personnel-records/1956/seg/firm/stylegan2_crops/pr',
	'font_test': '/mnt/data01/AWS_S3_CONTAINER/personnel-records/1956/seg/firm/stylegan2_crops_sample/pr',

	'font_gs_train': '/mnt/data01/AWS_S3_CONTAINER/personnel-records/1956/seg/firm/stylegan2_crops_grayscale/pr',
	'font_gs_test': '/mnt/data01/AWS_S3_CONTAINER/personnel-records/1956/seg/firm/stylegan2_crops_grayscale_sample/pr',

	'font_train_modern': '/mnt/data02/Japan/font_gen/paired_training_data/pr/labeled_validated_rendered_chars',
	'font_test_modern': '/mnt/data02/Japan/font_gen/paired_training_data/pr/rendered_chars_for_overfitting',
	'font_train_historical': '/mnt/data02/Japan/font_gen/paired_training_data/pr/labeled_validated_char_crops',
	'font_test_historical': '/mnt/data02/Japan/font_gen/paired_training_data/pr/char_crops_for_overfitting',
}

model_paths = {
	'ir_se50': 'pretrained_models/model_ir_se50.pth',
	'resnet34': '/mnt/workspace/pretrained_models/resnet34-333f7ec4.pth',
	'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pt',
	'stylegan_cars': 'pretrained_models/stylegan2-car-config-f.pt',
	'stylegan_church': 'pretrained_models/stylegan2-church-config-f.pt',
	'stylegan_horse': 'pretrained_models/stylegan2-horse-config-f.pt',
	'stylegan_ada_wild': 'pretrained_models/afhqwild.pt',
	'stylegan_toonify': 'pretrained_models/ffhq_cartoon_blended.pt',
	'shape_predictor': 'pretrained_models/shape_predictor_68_face_landmarks.dat',
	'circular_face': 'pretrained_models/CurricularFace_Backbone.pth',
	'mtcnn_pnet': 'pretrained_models/mtcnn/pnet.npy',
	'mtcnn_rnet': 'pretrained_models/mtcnn/rnet.npy',
	'mtcnn_onet': 'pretrained_models/mtcnn/onet.npy',
	'moco': '/mnt/workspace/pretrained_models/moco_v2_800ep_pretrain.pt' 
}
