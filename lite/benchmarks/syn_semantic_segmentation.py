from valor_lite.semantic_segmentation.benchmark import benchmark

if __name__ == "__main__":

    benchmark(
        bitmask_shape=(10000, 10000),
        number_of_images=10,
        number_of_unique_labels=10,
        memory_limit=4.0,
        time_limit=10.0,
        repeat=1,
        verbose=False,
    )
