# main.py
from processor import Processor

# from upper_processor import Processor

def main():
    # 可以自行加 argparse，这里只做演示
    base_dir = "DataSet"
    exercise_types = ["Squat2-raw"]
    sets = ["set1", "set2", "set3"]

    # 初始化 Processor；它会自动加载数据、模型、优化器
    processor = Processor(
        base_dir=base_dir,
        exercise_types=exercise_types,
        sets=sets,
        split_by_set = True,
        test_ratio=0.1,
        val_ratio=0.1,
        random_seed=42,
        use_gpu=True,
        device_id=0,
        batch_size=24,
        base_lr=1e-3,
        weight_decay=1e-4,
        num_epoch=2,
        optimizer_type='Adam',
        fusion_layer=0,
        n_hid_dec = 24,
        cross_w=0
    )

    # 启动训练 (内部会先 train 若干 epoch, 再 test)
    processor.run()
   
if __name__ == "__main__":
    main()
