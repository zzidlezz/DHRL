from layers import *
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class DHRL(nn.Module):
    def __init__(self, args):
        super(DHRL, self).__init__()


        self.image_dim = args.image_dim
        self.text_dim = args.text_dim

        self.img_hidden_dim = args.img_hidden_dim
        self.txt_hidden_dim = args.txt_hidden_dim
        self.common_dim = args.img_hidden_dim[-1]
        self.nbit = int(args.nbit)
        self.classes = args.classes
        self.batch_size = 0
        assert self.img_hidden_dim[-1] == self.txt_hidden_dim[-1]

        self.nhead = args.nhead
        self.act = args.trans_act
        self.dropout = args.dropout
        self.num_layer = args.num_layer

        self.imageMLP = Encoder(hidden_dim=self.img_hidden_dim, act=nn.Tanh())
        self.textMLP = Encoder(hidden_dim=self.txt_hidden_dim, act=nn.Tanh())

        self.imageConcept = nn.Linear(self.common_dim, self.common_dim * self.nbit)
        self.textConcept = nn.Linear(self.common_dim, self.common_dim * self.nbit)

        self.imagePosEncoder = PositionalEncoding(d_model=self.common_dim, dropout=self.dropout)
        self.textPosEncoder = PositionalEncoding(d_model=self.common_dim, dropout=self.dropout)

        imageEncoderLayer = TransformerEncoderLayer(d_model=self.common_dim,
                                                    nhead=self.nhead,
                                                    dim_feedforward=self.common_dim,
                                                    activation=self.act,
                                                    dropout=self.dropout)
        imageEncoderNorm = nn.LayerNorm(normalized_shape=self.common_dim)
        self.imageTransformerEncoder = TransformerEncoder(encoder_layer=imageEncoderLayer, num_layers=self.num_layer, norm=imageEncoderNorm)

        textEncoderLayer = TransformerEncoderLayer(d_model=self.common_dim,
                                                   nhead=self.nhead,
                                                   dim_feedforward=self.common_dim,
                                                   activation=self.act,
                                                   dropout=self.dropout)
        textEncoderNorm = nn.LayerNorm(normalized_shape=self.common_dim)
        self.textTransformerEncoder = TransformerEncoder(encoder_layer=textEncoderLayer, num_layers=self.num_layer, norm=textEncoderNorm)

        self.hash = nn.Sequential(
            nn.Conv2d(in_channels=self.nbit * self.common_dim, out_channels=self.nbit * self.common_dim // 2, kernel_size=1, groups=self.nbit),
            nn.BatchNorm2d(self.nbit * self.common_dim // 2),
            nn.Tanh(),
            nn.Conv2d(in_channels=self.nbit * self.common_dim // 2, out_channels=self.nbit, kernel_size=1, groups=self.nbit),
            nn.Tanh()
        )


        self.classify = nn.Linear(self.nbit, self.classes)


    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, image, text):
        self.batch_size = len(image)


        imageH = self.imageMLP(image)


        textH = self.textMLP(text)

        imageC = self.imageConcept(imageH).reshape(imageH.size(0), self.nbit, self.common_dim).permute(1, 0, 2) # (nbit, bs, dim)
        textC = self.textConcept(textH).reshape(textH.size(0), self.nbit, self.common_dim).permute(1, 0, 2) # (nbit, bs, dim)

        imageSrc = self.imagePosEncoder(imageC)
        textSrc = self.textPosEncoder(textC)

        imageMemory = self.imageTransformerEncoder(imageSrc)
        textMemory = self.textTransformerEncoder(textSrc)




        codeI = self.hash(imageMemory.permute(1, 0, 2).reshape(self.batch_size, self.nbit * self.common_dim, 1, 1)).squeeze()
        codeT = self.hash(textMemory.permute(1, 0, 2).reshape(self.batch_size, self.nbit * self.common_dim, 1, 1)).squeeze()


        return codeI,codeT, self.classify(codeI),self.classify(codeT)





