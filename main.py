from entitys.personSearch import *
from entitys.personSearchPose import *
import torch


class MainClass(BaseClass):
    def __init__(self):
        super().__init__() 

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.config_resnet50 = self.getConfig('resnet50')
        
        with torch.no_grad():
            # 使用单一特征根据文件夹query_folder中的人图来查询在视频中是否出现并按照文件名标出来id
            searcher = PersonSearch(
                weights=self.config_resnet50['weights'],
                device=self.device,
                img_size=int(self.config_resnet50['img_size']),
                conf_thres=float(self.config_resnet50['conf_thres']),
                match_threshold=float(self.config_resnet50['match_threshold']),
                query_folder=self.config_resnet50['query_folder']
            )
            # 使用综合特征根据文件夹query_folder中的人图来查询在视频中是否出现并按照文件名标出来id
            # searcher = PersonSearchPose(
            #     weights=self.config_resnet50['weights'],
            #     device=self.device,
            #     img_size=int(self.config_resnet50['img_size']),
            #     conf_thres=float(self.config_resnet50['conf_thres']),
            #     match_threshold=float(self.config_resnet50['match_threshold']),
            #     reid_weight=float(self.config_resnet50['reid_weight']),
            #     pose_weight=float(self.config_resnet50['pose_weight']),
            #     color_weight=float(self.config_resnet50['color_weight']),
            #     query_folder=self.config_resnet50['query_folder']
            # )

            
            searcher.search_video(self.config_resnet50['source'], self.config_resnet50['view_img'], self.config_resnet50['save_path'])


if __name__ == '__main__':
    model = MainClass()
