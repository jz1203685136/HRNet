def load_data(self, index):
    fn = self.fns[index]
    m_img = cv2.imread(join(self.datadir, fn))

    # print(self.imgsize)
    assert self.imgsize in ['middle', 'small', 'origin']
    if self.imgsize == 'middle':
        if m_img.shape[0] < m_img.shape[1]:
            size = (int(512 * m_img.shape[1] / m_img.shape[0]), 512)
        else:
            size = (512, int(512 * m_img.shape[0] / m_img.shape[1]))
    else:
        if m_img.shape[0] < m_img.shape[1]:
            size = (int(256 * m_img.shape[1] / m_img.shape[0]), 256)
        else:
            size = (256, int(256 * m_img.shape[0] / m_img.shape[1]))
    if not (m_img.shape[0] == size[1] and m_img.shape[1] == size[0]):
        scale = int(math.log2(min(m_img.shape[0] / size[1], m_img.shape[1] / size[0])))
        for i in range(0, scale):
            m_img = cv2.pyrDown(m_img)
        if not (m_img.shape[0] == size[1] and m_img.shape[1] == size[0]):
            m_img = cv2.resize(m_img, size, cv2.INTER_AREA)

    m_img = cv2.cvtColor(m_img, cv2.COLOR_BGR2RGB)
    M = np.transpose(np.float32(m_img) / 255.0, (2, 0, 1))
    data = {'input': M, 'target_t': torch.zeros([1, 0]), 'fn': fn[:-4], 'mask': torch.zeros([1, 0])}
    return data