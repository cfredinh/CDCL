import pdb
import h5py
import skimage
from utils import *
from .bases import BaseSet
from scipy.io import mmread
from collections import Counter
from sklearn.model_selection import train_test_split
from torchvision.transforms import ToTensor, ToPILImage


DATA_INFO = {
              "RxRx1_HUVEC": {"dataset_location": "RXRX1"},
              "CPG0004_full_img_large": {"dataset_location": "CPG0004_full_img_large"}
}


class DomainSet(BaseSet):
    
    def get_subset(self, dataframe, subset_size=10000, overwrite=False):
        dtvs_dir = os.path.join(self.root_dir, "domain_trainval_splits", "subset_splits")
        check_dir(dtvs_dir)        
        subset_path = os.path.join(dtvs_dir, f"{self.name}-test_domain_{self.test_domain}-{self.mode}_subset-{subset_size}.csv")

        if not os.path.isfile(subset_path) or overwrite:
            if is_rank0():
                print("Creating a new subset split for \"{}\" !".format(self.name))
                _, subset = train_test_split(dataframe, test_size=subset_size) 
                subset.to_csv(subset_path)
            synchronize()
            subset = pd.read_csv(subset_path)

        else:
            subset = pd.read_csv(subset_path)
            if len(subset) != subset_size:
                print_ddp(f"Found updated subset size for \"{dataset_name}\" for \"{dataset_name}\"  of size {subset_size} !")
                self.get_subset(dataframe, subset_size=subset_size, overwrite=True)
        
        return subset
    
    def get_train_val_splits(self, dataframe, val_size=0.1, overwrite=False, allowed_div=1):
        dtvs_dir = os.path.join(self.root_dir, "domain_trainval_splits")
        check_dir(dtvs_dir)
        all_doms = deepcopy(self.DNET_DOMAINS)
        
        train_data, val_data = [], []
        for didx in range(len(all_doms)):
            domname = all_doms[didx]
            dom_data = dataframe[dataframe.domain.isin([all_doms[didx]])].copy()
            
            dom_train_path = os.path.join(dtvs_dir, f"{self.name}_{domname}_train.csv")
            dom_val_path   = os.path.join(dtvs_dir, f"{self.name}_{domname}_val.csv")
            
            total_size     = len(dom_data)
            int_val_size   = int(total_size * val_size)
            int_train_size = int(total_size - int_val_size)
            
            if not os.path.isfile(dom_train_path) or not os.path.isfile(dom_val_path) or overwrite:
                if is_rank0():
                    print("Creating a new train/val split for \"{}-{}\" !".format(self.name, domname))
                    dom_train, dom_val = train_test_split(dom_data, test_size=val_size) 
                    dom_train.to_csv(dom_train_path)
                    dom_val.to_csv(dom_val_path)
                synchronize()
                dom_train = pd.read_csv(dom_train_path)
                dom_val   = pd.read_csv(dom_val_path)                    
                    
            else:
                dom_train = pd.read_csv(dom_train_path)
                dom_val   = pd.read_csv(dom_val_path)
                
                train_split_ok = int_train_size - allowed_div <= len(dom_train) <= int_train_size + allowed_div 
                val_split_ok   = int_val_size   - allowed_div <= len(dom_val)   <= int_val_size   + allowed_div 
                if not (train_split_ok and val_split_ok):
                    print_ddp(f"Found updated train/validation size for \"{dataset_name}\" !")
                    self.get_train_val_splits(dataframe, val_size=val_size, overwrite=True)
            if domname in self.domains:
                train_data.append(dom_train)
                val_data.append(dom_val)
                
                
        assert len(train_data) == len(val_data) == self.num_domains - 1, "Mismatch of train/val/test domains"
        train_data = pd.concat(train_data).reset_index(drop=True) # 
        val_data   = pd.concat(val_data).reset_index(drop=True)   #
        
        return train_data, val_data            
    
    def get_data_as_list(self, use_subset=0):
        data_list = []
        df = self.get_dataframe()
        
        coresponding_labels_and_names = df[["label","label_name"]].drop_duplicates()
        self.int_to_labels = dict(zip(coresponding_labels_and_names.label, coresponding_labels_and_names.label_name))
        self.labels_to_int = {val: key for key, val in self.int_to_labels.items()}
        
        if self.domain_wise_test:
            if self.mode != 'test':
                train_data, val_data = self.get_train_val_splits(df, val_size=0.1)
                data = train_data if self.mode == 'train' else val_data
            else:
                data = df[df.domain == self.test_domain]
        else:
            data = df
            
        if use_subset and len(data) > use_subset:
            data = self.get_subset(data, subset_size=use_subset)            
                
        self.df = data
        labels    = data['label'].values.tolist()
        img_paths = data['img_path'].values.tolist()
        domain    = data['domain'].values.tolist()
        df_index  = data.index.values.tolist()
        
        domains_found, counts_in_each_domain = np.unique(domain, return_counts=True)
        
        if is_rank0():
            print(domains_found, counts_in_each_domain)
        
        examples_in_each_domain_dict = {self.domain_to_int[dom] : dom_count for dom, dom_count in zip(domains_found, counts_in_each_domain)}

        examples_in_each_domain = np.zeros(len(self.DNET_DOMAINS))
        
        for k,v in examples_in_each_domain_dict.items():
            examples_in_each_domain[k] = v

        self.examples_in_each_domain = examples_in_each_domain

        if is_rank0():
            print(self.examples_in_each_domain)        
        
        if self.mode == 'train' and not self.fb:
            self.include_info = False
        else:
            self.include_info = False
        
        self.data_id_2_df_id = {i: val   for i,  val in enumerate(df_index)}
        self.df_id_2_data_id = {val: key for key,val in self.data_id_2_df_id.items()}
        
        data_list = [{'img_path': os.path.join(self.root_dir,  str(img_path)), 
                       'label': label, 'domain': self.domain_to_int[dom], 'dataset': self.name}
                     for img_path, label, dom in zip(img_paths, labels, domain)]
        
        return data_list   
    
    def __getitem__(self, idx): 
        
        img_path = self.data[idx]['img_path']
        cls    = torch.as_tensor(self.data[idx]['label'])
        domain = torch.as_tensor(self.data[idx]['domain'])
        label  = (cls,domain)
        if self.include_info:
            label  = (cls,domain,cls)
        
        img = self.get_image(img_path)
        
        if self.cross_batch_training and self.mode == 'train' and not self.fb:
            df_idx = self.data_id_2_df_id[idx]
            id_info = self.df.loc[df_idx][["domain","label"]]
            
            df_id_sample = int(self.df[(self.df.domain != id_info[0]) & (self.df.label == id_info[1])].sample().index.values)
            
            idx_2 = self.df_id_2_data_id[df_id_sample]
            
            img_path_2 = self.data[idx_2]['img_path']
            cls_2      = torch.as_tensor(self.data[idx_2]['label'])
            domain_2   = torch.as_tensor(self.data[idx_2]['domain'])
            label      = (cls,(domain,domain_2))
            if self.include_info:
                label  = (cls,(domain,domain_2),cls)

            img_2 = self.get_image(img_path_2)

        if self.resizing is not None:
            img = self.resizing(img)
            if self.cross_batch_training and self.mode == 'train' and not self.fb:
                img_2 = self.resizing(img_2)
        
        if self.transform is not None:
            if isinstance(self.transform, list):
                if self.cross_batch_training and self.mode == 'train'  and not self.fb:
                    img_list = [self.transform[0](img),self.transform[1](img_2)]
                    img_list += ([tr(img) for tr in self.transform[2:5]])
                    img_list += ([tr(img_2) for tr in self.transform[5:]])
                    img   = img_list
                else:
                    img = [tr(img) for tr in self.transform]
                
            else:
                if self.is_multi_crop:
                    img = self.multi_crop_aug(img, self.transform)
                else:
                    img = [self.transform(img) for _ in range(self.num_augmentations)]
            img = img[0] if len(img) == 1 and isinstance(img, list) else img   
        
        return img, label
    
    def get_image(self,img_path):
        png_path = '.'.join(img_path.split('.')[:-1]) + '.png'
        if os.path.exists(png_path):
            img = self.get_x(png_path)
            img_path = png_path
        else:
            img = self.get_x(img_path)
        return img
    
    def get_dataframe(self):
        
        if is_rank0():
            print("Test domain: ", self.test_domain, " Training domains: ", self.domains)

        combined_list = []
        combined_df_list = []
        
        for domain in self.DNET_DOMAINS:
            
            mypath = self.root_dir + "/" + domain + "/"
            classes = os.listdir(mypath)
            for cls in classes:
                cls_path  = os.path.join(self.root_dir, domain, cls)
                img_paths = os.listdir(cls_path)
                temp_df = pd.DataFrame(img_paths, columns = ["path"])
                temp_df["domain"] = domain
                temp_df["label_name"] = cls
                temp_df["path"]   = [os.path.join(domain, cls, p) for p in temp_df["path"]]
                temp_df["img_path"]   = temp_df["path"]
                combined_df_list.append(temp_df)
        
        combined = pd.concat(combined_df_list).reset_index(drop=True)
        unique_ = combined.label_name.unique()
        unique_.sort()
        translate = {x:i for i,x in enumerate(unique_)}
        combined["label"] = combined["label_name"].map(translate)
        
        return combined    
    
    def get_stats(self):
        
        stat_save_path = os.path.join(self.root_dir, "domain_stats.pickle")   
        stats = load_pickle(stat_save_path)
        
        mean = []
        std = []
        for dom in self.DNET_DOMAINS:
            if dom not in self.domains and self.mode != 'test': continue              
            if dom != self.test_domain and self.mode == 'test': continue        
            mean.append(stats[dom]['mean'])
            std.append(stats[dom]['std'])
        
        self.mean = np.mean(mean, axis=0)
        self.std = np.mean(std, axis=0)

    

    
    
class RxRx1_HUVEC(DomainSet):
    
    name = "RxRx1_HUVEC"
    img_channels = 6
    is_multiclass = True
    cross_batch_training = False
    include_info = True
    task = 'classification'
    normalize_per_plate = True
    max_pixel_value = 255.0
    drop_treatment_duplicates = False
    drop_treatment_duplicates_keep_controlls = False



    mean = (0.02051845, 0.07225967, 0.0303462 , 0.03936887, 0.01122004, 0.03018961)
    std  = (0.02233115, 0.0472001 , 0.01045581, 0.02445364, 0.00950606, 0.01063623)
    
    split_number = 1

    h5_path_x = "/storage_fast/user/datasets/RXRX1/rxrx1/RXRX1_large_plate_info_corrected_size_large_all_data_in_one.hdf5"

    plate_wise_control_path = "/storage/user/datasets/RXRX1/rxrx1/plate_negative_control_norm.csv"
    
    domain_wise_test = False
    DNET_DOMAINS = [x for x in range(24)]
    

    sub_sample       = False
    subset_data      = False
    subset_strategy  = ""
    

    int_to_labels = {x: int(x) for x in range(1139)}
    int_to_domain = {i: int(x) for i, x in enumerate(DNET_DOMAINS)}
    labels_to_int = {val: key for key, val in int_to_labels.items()}
    domain_to_int = {val: key for key, val in int_to_domain.items()}

    
    target_metric = 'accuracy'
    knn_nhood = 10    
    n_classes = len(int_to_labels)

    int_to_id = int_to_labels
    
    def __init__(self, dataset_params, mode='train', fb = False, use_subset=0):
        
        self.use_subset = use_subset        
        self.attr_from_dict(dataset_params)
        self.dataset_location = DATA_INFO["RxRx1_HUVEC"]["dataset_location"]
        self.root_dir = os.path.join(self.data_location, self.dataset_location)
        
        self.test_domain = self.data_test_domain
        self.domains  = deepcopy(self.DNET_DOMAINS)
        
        self.num_domains = len(self.domains)
        
        self.mode = mode
        self.read_mode = "train"
        self.fb   = fb
        self.get_stats()
        self.data = self.get_data_as_list(use_subset=self.use_subset)
        self.transform, self.resizing = self.get_transforms(albumentations=True)
        
        self.h5_file_x = None 

        if mode == "test":
            self.return_id = True
        else:
            self.return_id = False
        
        
    def get_dataframe(self):        
        
        
        with h5py.File(self.h5_path_x, 'r') as full_data:
            
            h5_file_y = np.array(full_data[self.read_mode + '_labels'])
            h5_file_b = np.array(full_data[self.read_mode + '_batch' ])
            h5_file_p = np.array(full_data[self.read_mode + '_plate' ])
            length = len(h5_file_y)
            df = pd.DataFrame({'label_name': h5_file_y[:], 'label': h5_file_y[:], 'plate': h5_file_p[:], 'domain': h5_file_b[:]}, columns=['label_name', 'label', 'plate', 'domain'])
            
            df["img_path"] = df.index

        all_domains = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        splits =   {1: {"train":all_domains[ 0:16],                    "eval": all_domains[16:20], "test": all_domains[20:24]},
                    2: {"train":all_domains[ 4:20],                    "eval": all_domains[20:24], "test": all_domains[ 0: 4]},
                    3: {"train":all_domains[ 8:24],                    "eval": all_domains[ 0: 4], "test": all_domains[ 4: 8]},
                    4: {"train":all_domains[12:24]+all_domains[ 0: 4], "eval": all_domains[ 4: 8], "test": all_domains[ 8:12]},
                    5: {"train":all_domains[16:24]+all_domains[ 0: 8], "eval": all_domains[ 8:12], "test": all_domains[12:16]},
                    6: {"train":all_domains[20:24]+all_domains[ 0:12], "eval": all_domains[12:16], "test": all_domains[16:20]}}

        df = df[df.domain.isin(splits[self.split_number][self.mode])]    
        print(splits[self.split_number][self.mode])
    
        if self.mode == "train" and not self.fb:
            if self.subset_data:
                if self.subset_strategy == "only_controls":

                    df = df[~(df.label_name < 1108)] # Only select controls for training
                elif self.subset_strategy == "only_treatments":
                    df = df[(df.label_name < 1108)]  # Only select treatments for training
                elif self.subset_strategy == "only_treatments_half_as_many":
                    df = df[(df.label_name < 1108)].drop_duplicates(subset=["label","plate"])  # Only select treatments for training
                elif self.subset_strategy == "only_treatments_half_as_many_with_controls":
                    unint = df[(df.label_name < 1108)].drop_duplicates(subset=["label","plate"]).index # Keeps all the controlls and one replicate per plate and label

                    print("NUMBER OF DROPED INDEXES", len(unint))

                    df    = df.drop(unint)
                    print("SIZE AFTER DROPPING", df.shape)

                elif self.subset_strategy == "only_half_as_many_treatments_with_controls":
                    drop_ids = [i for i in df[(df.label_name < 1108)].label_name.unique() if i%2==0]
                    df = df[~df.label_name.isin(drop_ids)]

        if self.drop_treatment_duplicates:
            df = df[~(df.label_name < 1108)].drop_duplicates(subset=["label","plate"])
        if self.drop_treatment_duplicates_keep_controlls:
            unint = df[(df.label_name < 1108)].drop_duplicates(subset=["label","plate"]).index # Keeps all the controlls and one replicate per plate and label
            df    = df.drop(unint)

        if is_rank0():    
            print("MAXIMUM PLATE NUMBER: ", df.plate.unique())
        
        return df
    
    def get_stats(self):
        
        stat_save_path = os.path.join(self.root_dir, "domain_stats.pickle")   
        
        mean = [1.0,1.0,1.0,1.0,1.0,1.0]
        std  = [1.0,1.0,1.0,1.0,1.0,1.0]
        
        self.mean = np.mean(mean, axis=0)*0.0
        self.std  = np.mean(std, axis=0)
        
        self.norm_df = pd.read_csv(self.plate_wise_control_path)
        
        self.mean_columns = ["mean_" + str(x) for x in range(6)]
        self.std_columns  = ["std_"  + str(x) for x in range(6)]
        
    def get_image(self,img_path):
        
        if self.h5_file_x == None:
            self.h5_file_x     = h5py.File(self.h5_path_x, 'r')[self.read_mode + '_img']
            self.h5_file_plate = h5py.File(self.h5_path_x, 'r')[self.read_mode + '_plate']
        
        img   = self.h5_file_x[int(img_path.split("/")[-1])]
        plate = self.h5_file_plate[int(img_path.split("/")[-1])]
        
        return img, plate
    
    def plate_normalize_image(self,x,p):
        x = x.float()
        if self.normalize_per_plate:
            
            plate_stats = self.norm_df[self.norm_df.plate == p]
            
            mean_vals = np.array([plate_stats[self.mean_columns].values])*(1/255.0)
            std_vals  = np.array([plate_stats[self.std_columns ].values])*(1/255.0) # Change this to epsilon
            
            norm_values_mean = torch.tensor(mean_vals).view(-1,1,1).type(torch.FloatTensor)
            norm_values_std  = torch.tensor(std_vals ).view(-1,1,1).type(torch.FloatTensor)
            
            x = (x-norm_values_mean)/(norm_values_std)
        else:
            
            mean_vals = np.array([[5.23220486,  18.42621528, 7.73828125,  10.0390625 ,   2.86111111,   7.69835069]])*(1/255.0)
            std_vals  = np.array([[5.69444444,  12.03602431, 2.66623264,   6.23567708,   2.42404514,   2.71223958]])*(1/255.0)
            
            norm_values_mean = torch.tensor(mean_vals).view(-1,1,1).type(torch.FloatTensor)
            norm_values_std  = torch.tensor(std_vals ).view(-1,1,1).type(torch.FloatTensor)
            
            x = (x-norm_values_mean)/(norm_values_std)
            
        return x
    
    def __getitem__(self, idx): 
        
        img_path = self.data[idx]['img_path']
        cls    = torch.as_tensor(self.data[idx]['label'])
        domain = torch.as_tensor(self.data[idx]['domain'])
        label  = (cls,domain)
        if self.include_info:
            label  = (cls,domain,cls)
        
        img, plate = self.get_image(img_path)

        if self.return_id:
            label  = (cls,domain, cls)
        
        if self.cross_batch_training and self.mode == 'train' and not self.fb:
            df_idx = self.data_id_2_df_id[idx]
            id_info = self.df.loc[df_idx][["domain","label"]]
            
            df_id_sample = int(self.df[(self.df.domain != id_info[0]) & (self.df.label == id_info[1])].sample().index.values)
            
            idx_2 = self.df_id_2_data_id[df_id_sample]
            
            img_path_2 = self.data[idx_2]['img_path']
            cls_2      = torch.as_tensor(self.data[idx_2]['label'])
            domain_2   = torch.as_tensor(self.data[idx_2]['domain'])
            label      = (cls,(domain,domain_2))
            if self.include_info:
                label  = (cls,(domain,domain_2), cls)

            img_2, plate_2 = self.get_image(img_path_2)
        elif (not self.cross_batch_training) and self.mode == 'train' and not self.fb:
            label      = (cls,(domain,domain))
            
        if self.resizing is not None:
            img = self.resizing(image=img)["image"]
            
            if self.cross_batch_training and self.mode == 'train' and not self.fb:
                img_2 = self.resizing(image=img_2)["image"]
                
            
                
        if self.transform is not None:
            
            if isinstance(self.transform, list):
                
                if self.cross_batch_training and self.mode == 'train'  and not self.fb:
                    img_list = [self.plate_normalize_image((self.transform[0](image=img   )["image"]), plate),
                                self.plate_normalize_image((self.transform[1](image=img_2 )["image"]), plate_2)]
                    img_list += ([self.plate_normalize_image(tr(image=img)["image"],   plate)   for tr in self.transform[2:5]])
                    img_list += ([self.plate_normalize_image(tr(image=img_2)["image"], plate_2) for tr in self.transform[5:]])
                    img   = img_list
                else:
                    img = [self.plate_normalize_image(tr(image=img)["image"], plate) for tr in self.transform]
                
            else:
                if self.is_multi_crop:
                    
                    img = self.multi_crop_aug(img, self.transform)
                else:
                    
                    img = [self.plate_normalize_image(self.transform(image=img)["image"], plate) for _ in range(self.num_augmentations)]                    
            
            img = [im.float() for im in img]        
            img = img[0] if len(img) == 1 and isinstance(img, list) else img  
        
        

        if img[0].dtype != torch.float32 or img[1].dtype != torch.float32:
            print(img[0].dtype)

        return img, label
    
    
    

class CPG0004_full_img_large(BaseSet):
    
    img_channels  = 5
    is_multiclass = True
    include_info  = False
    return_id     = True
    task = 'classification'    
    cross_batch_training = True
    
    mean = (14.61494155, 28.87414807, 32.61800756, 36.41028929, 22.27769592)
    std  = (28.80363038, 31.39763568, 32.06969697, 32.35857192, 25.8217434)
    max_pixel_value = 255.0
    int_to_labels   = {i: str(i+1) for i in range(9412)}
    num_domains     = 136
    normalize_per_plate = True
    
    subset_test     = False
    predict_moa     = False

    cross_validation = True
    split_number     = 1
    
    if predict_moa:
        int_to_labels = {i: str(i+1) for i in range(50)}
    
    target_metric   = 'accuracy'
    knn_nhood       = 10  
    n_classes       = len(int_to_labels)
    
    labels_to_int = {val: key for key, val in int_to_labels.items()}
    int_to_id = int_to_labels
    
    def __init__(self, dataset_params, fb = False, mode='train', use_subset=0):


        self.attr_from_dict(dataset_params)
        self.dataset_location = DATA_INFO["CPG0004_full_img_large"]["dataset_location"]
        
       

        self.fb = fb

        self.root_dir = '/storage_fast/user/datasets/large-Cell-Painting-Dataset-IMAGE-DATA/CPG0004-large/CPG0004-270/'
        self.root_dir = '/storage/user/datasets/large-Cell-Painting-Dataset-IMAGE-DATA/CPG0004-large/'
        self.img_size = 270
        self.img_size = 1080
        
        self.mode = mode
        self.get_stats()
        self.data = self.get_data_as_list()
        self.transform, self.resizing = self.get_transforms(albumentations=True)
        
        if mode == "train" and not fb:
            self.return_id = False
        else:
            self.return_id = True
        
        print("CROSS BATCH TRAINING SET TO: ", self.cross_batch_training)
        
    def get_data_as_list(self):

        data_list = []
        
        path = "/storage/user/datasets/large-Cell-Painting-Dataset/datasets/CPG0004/top50_moa_with_dmso.csv"
        
        datainfo = pd.read_csv(path, index_col=0, engine='python')
        
        
        domainlist = datainfo.Metadata_Plate_ID.values
        moa_list   = datainfo.Metadata_moa_ID.values
        labellist  = datainfo.Treatment_ID.tolist()
        img_names  = datainfo.combined_paths.tolist()
        
        if self.predict_moa:
            labellist   = datainfo.Metadata_moa_ID.values.tolist()
        
        if self.subset_test:
            moa_list   = datainfo.index.values
        

        if self.cross_validation:
            split = datainfo.replicate_ID.tolist()
            
            dataframe  = pd.DataFrame(list(zip(img_names, labellist, domainlist, moa_list, split)), 
                                    columns=['img_path', 'label', "domains", "moa", "split"])

            
            splits =   {1: {"train":[1., 2., 3.], "eval": [4.], "test": [5.]},
                        2: {"train":[2., 3., 4.], "eval": [5.], "test": [1.]},
                        3: {"train":[3., 4., 5.], "eval": [1.], "test": [2.]},
                        4: {"train":[1., 4., 5.], "eval": [2.], "test": [3.]},
                        5: {"train":[1., 2., 5.], "eval": [3.], "test": [4.]}}
            

            train_ids = dataframe[dataframe.split.isin(splits[self.split_number]["train"])].index.values
            val_ids   = dataframe[dataframe.split.isin(splits[self.split_number]["eval"] )].index.values
            test_ids  = dataframe[dataframe.split.isin(splits[self.split_number]["test"] )].index.values

            if self.mode == 'train':
                data = dataframe.loc[train_ids,:]
            elif self.mode in ['val', 'eval']:
                data = dataframe.loc[val_ids,  :]
            else:
                data = dataframe.loc[test_ids, :]
        
        else:

            split      = datainfo.Split.tolist()
            
            dataframe  = pd.DataFrame(list(zip(img_names, labellist, domainlist, moa_list, split)), 
                                    columns=['img_path', 'label', "domains", "moa", "split"])
        
            train_ids = dataframe[dataframe.split == "Training"].index.values
            val_ids   = dataframe[dataframe.split == "Validation"  ].index.values
            test_ids  = dataframe[dataframe.split == "Test" ].index.values

            if self.mode == 'train':
                data = dataframe.loc[train_ids,:]
            elif self.mode in ['val', 'eval']:
                data = dataframe.loc[val_ids,  :]
            else:
                data = dataframe.loc[test_ids, :]


        print("NUMBER OF UNIQUE LABELS in :", self.mode, data['label'   ].nunique())

        labels    = data['label'   ].values.tolist()
        domains   = data['domains' ].values.tolist()
        img_paths = data['img_path'].values.tolist()
        moa_data  = data['moa'     ].values.tolist()

        
        domains_found, counts_in_each_domain = np.unique(domains, return_counts=True)
        
        
        examples_in_each_domain_dict = {dom : dom_count for dom, dom_count in zip(domains_found, counts_in_each_domain)}

        examples_in_each_domain = np.zeros(self.num_domains) #np.zeros(len(domains_found)+1)
        if is_rank0():
            print("examples_in_each_domain_dict ", examples_in_each_domain_dict)
        for k,v in examples_in_each_domain_dict.items():
            examples_in_each_domain[k] = v
        
        
        self.examples_in_each_domain = examples_in_each_domain  
        
        data_list = [{'img_path': img_path, 'label': label, "domain": domain, 'dataset': self.name, 'moa': moa}
                     for img_path, label, domain, moa in zip(img_paths, labels, domains, moa_data)]
        
        self.df = pd.DataFrame(data_list)
        
        return data_list    

    def get_stats(self):
        
        path = "/storage/user/datasets/large-Cell-Painting-Dataset/datasets/CPG0004/top20_moa_plate_norms.csv"
        datainfo = pd.read_csv(path, index_col=0, engine='python')
        
        print("GETTING STATS")
        
        mean = [1.0,1.0,1.0,1.0,1.0]
        std  = [1.0,1.0,1.0,1.0,1.0]
        
        self.mean = np.mean(mean, axis=0)*0.0
        self.std  = np.mean(std, axis=0)
        
        
        self.norm_df = datainfo
        self.norm_df["plate"] = self.norm_df.index
        
        self.mean_columns = ["DNA_mean", "ER_mean", "RNA_mean", "AGP_mean", "Mito_mean"]
        self.std_columns  = ["DNA_std",  "ER_std",  "RNA_std",  "AGP_std",  "Mito_std"]
    
    def plate_normalize_image(self,x,p):
        x = x.float()
        if self.normalize_per_plate:
            
            plate_stats = self.norm_df[self.norm_df.plate == p]
            
            mean_vals = (plate_stats[self.mean_columns].values)*(1/255.0)
            std_vals  = (plate_stats[self.std_columns ].values )*(1/255.0) 
            
            norm_values_mean = torch.tensor(np.array([mean_vals])).view(-1,1,1).type(torch.FloatTensor)
            norm_values_std  = torch.tensor(np.array([std_vals ])).view(-1,1,1).type(torch.FloatTensor)
            
            x = (x-norm_values_mean)/(norm_values_std)
        else:
            
            mean_vals = np.array([14.61494155, 28.87414807, 32.61800756, 36.41028929, 22.27769592])*(1/255.0)
            std_vals  = np.array([28.80363038, 31.39763568, 32.06969697, 32.35857192, 25.8217434])*(1/255.0)
            
            norm_values_mean = torch.tensor(np.array([mean_vals])).view(-1,1,1).type(torch.FloatTensor)
            norm_values_std  = torch.tensor(np.array([std_vals ])).view(-1,1,1).type(torch.FloatTensor)
            
            x = (x-norm_values_mean)/(norm_values_std)
            
        return x
    
    def get_image(self,img_paths):
        
       
        full = np.zeros((self.img_size, self.img_size,5))

        for i, path in enumerate(img_paths.split(",")):
            full[:,:,i] = skimage.io.imread(self.root_dir+path)

        
        return full 
    
    def __getitem__(self, idx): 
        
        img_path = self.data[idx]['img_path']
        label    = torch.as_tensor(self.data[idx]['label'])
        domain   = torch.as_tensor(self.data[idx]['domain'])
        moa      = torch.as_tensor(self.data[idx]['moa'])
        plate    = self.data[idx]['domain']
        
        img = self.get_image(img_path)
        img = img.astype(np.float32)
        

        
        if self.return_id:
            label  = (label,domain, moa)
            
        if self.cross_batch_training and self.mode == 'train' and not self.fb:
            
            
            if (self.df[(self.df.domain != int(domain)) & (self.df.label == int(label))]).shape[0] > 0:
                df_id_sample = int(self.df[(self.df.domain != int(domain)) & (self.df.label == int(label))].sample().index.values)
            else:
                df_id_sample = int(self.df[(self.df.label == int(label))].sample().index.values)

            idx_2 = df_id_sample
            
            img_path_2 = self.data[idx_2]['img_path']
            cls_2      = torch.as_tensor(self.data[idx_2]['label'])
            plate_2    = self.data[idx_2]['domain']
            domain_2   = torch.as_tensor(self.data[idx_2]['domain'])
            
            label      = (label,(domain,domain_2))
            
            img_2 = self.get_image(img_path_2)
            img_2 = img_2.astype(np.float32)
            
        elif (not self.cross_batch_training) and self.mode == 'train' and not self.fb:
            label      = (label,(domain,domain))
            
        if self.resizing is not None:
            img = self.resizing(image=img)["image"]
            
            if self.cross_batch_training and self.mode == 'train' and not self.fb:
                img_2 = self.resizing(image=img_2)["image"]
                
            
                
        if self.transform is not None:
            
            if isinstance(self.transform, list):
                
                if self.cross_batch_training and self.mode == 'train'  and not self.fb:
                    img_list = [self.plate_normalize_image((self.transform[0](image=img   )["image"]), plate),
                                self.plate_normalize_image((self.transform[1](image=img_2 )["image"]), plate_2)]
                    img_list += ([self.plate_normalize_image(tr(image=img)["image"],   plate)   for tr in self.transform[2:5]])
                    img_list += ([self.plate_normalize_image(tr(image=img_2)["image"], plate_2) for tr in self.transform[5:]])
                    img   = img_list
                else:
                    img = [self.plate_normalize_image(tr(image=img)["image"], plate) for tr in self.transform]
                
            else:
                if self.is_multi_crop:
                    
                    img = self.multi_crop_aug(img, self.transform)
                else:
                    
                    img = [self.plate_normalize_image(self.transform(image=img)["image"], plate) for _ in range(self.num_augmentations)]                    
            
            img = [im.float() for im in img]        
            img = img[0] if len(img) == 1 and isinstance(img, list) else img  
        

        if img[0].dtype != torch.float32 or img[1].dtype != torch.float32:
            print(img[0].dtype)


        return img, label
    
    
