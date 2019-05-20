from VSR.RBPN.youku import YoukuDataset


def main():  # for test
    ykd = YoukuDataset('dataset/Youku_00000', 4, True, 0, True)
    ykd.__getitem__(0)
    return


if __name__ == '__main__':
    main()
