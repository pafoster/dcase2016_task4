# DCASE 2016::Domestic Audio Tagging / Baseline System
# Copyright (C) 2016 Peter Foster (p.a.foster@qmul.ac.uk) / QMUL
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from dataset import *

class CHiMEHome_DomesticAudioTag_DevelopmentSet(Dataset):
    def __init__(self, data_path=None, name='CHiMEHome-audiotag-development',evaluation_folds=5):
        self.name = name

        self.authors = 'Peter Foster, Siddharth Sigtia, Sacha Krstulovic, Jon Barker, and Mark Plumbley'
        self.name_remote = 'The CHiME-Home dataset is a collection of annotated domestic environment audio recordings.'
        self.url = ''
        self.audio_source = 'Field recording'
        self.audio_type = 'Natural'
        self.recording_device_model = 'Unknown'
        self.microphone_model = 'Unknown'

        self.sampling_rate = '16kHz'
        self.evaluation_folds = evaluation_folds

        Dataset.__init__(self, data_path=data_path)

        self.package_list = [
            {
                'remote_package': 'https://archive.org/download/chime-home/chime_home.tar.gz',
                'local_package': os.path.join(self.local_path, 'chime_home.tar.gz'),
                'local_audio_path': os.path.join(self.local_path, 'chime_home', 'chunks'),
                'development_chunks_refined_csv': os.path.join(self.local_path, 'chime_home', 'development_chunks_refined.csv'),
                'development_chunks_refined_crossval_csv': os.path.join(self.local_path, 'chime_home', 'development_chunks_refined_crossval_dcase2016.csv'),
            },
        ]

    @property
    def audio_files(self):
        """
        Get all audio files in the dataset, use only file from CHime-Home-refined set.
        :return: file list with absolute paths
        """        
        if self.files is None:
            refined_files = []
            with open(self.package_list[0]['development_chunks_refined_csv'], 'rt') as f:
                for row in csv.reader(f, delimiter=','):
                    refined_files.append(row[1])
            if 'evaluation_chunks_refined_csv' in self.package_list[0].keys():
                 with open(self.package_list[0]['evaluation_chunks_refined_csv'], 'rt') as f:
                    for row in csv.reader(f, delimiter=','):
                        refined_files.append(row[1])       

            self.files = []
            for file in self.package_list:
                path = file['local_audio_path']
                if path:
                    l = os.listdir(path)
                    p = path.replace(self.local_path + os.path.sep, '')
                    for f in l:
                        fileName, fileExtension = os.path.splitext(f)
                        fileName, samplingRate = os.path.splitext(fileName)
                        if fileExtension[1:] in self.audio_extensions and fileName in refined_files and samplingRate[1:] in self.sampling_rate:
                            self.files.append(os.path.abspath(os.path.join(path, f)))

            self.files.sort()
        return self.files

    def read_chunk_meta(self, meta_filename):
        if os.path.isfile(meta_filename):
            meta_file_handle = open(meta_filename, 'rt')
            try:
                meta_file_reader = csv.reader(meta_file_handle, delimiter=',')
                data = {}
                for meta_file_row in meta_file_reader:
                    data[meta_file_row[0]] = meta_file_row[1]
            finally:
                meta_file_handle.close()
            return data
        else:
            return None

    def tagcode_to_taglabel(self, tag):
        map = {'c': 'child speech',
               'm': 'adult male speech',
               'f': 'adult female speech',
               'v': 'video game/tv',
               'p': 'percussive sound',
               'b': 'broadband noise',
               'o': 'other',
               'S': 'silence/background',
               'U': 'unidentifiable'
               }
        if tag in map:
            return map[tag]
        else:
            return None            

    def on_after_extract(self):
        # Make legacy dataset compatible with DCASE2016 dataset scheme
        if not os.path.isfile(self.meta_file):
            section_header('Generating meta file for dataset')

            scene_label = 'home'
            f = open(self.meta_file, 'wt')
            try:
                writer = csv.writer(f, delimiter='\t')
                for file in self.audio_files:
                    raw_path, raw_filename = os.path.split(file)
                    relative_path = self.absolute_to_relative(raw_path)

                    base_filename, file_extension = os.path.splitext(raw_filename)
                    base_filename, sampling_rate = os.path.splitext(base_filename)
                    annotation_filename = os.path.join(raw_path, base_filename + '.csv')
                    meta_data = self.read_chunk_meta(annotation_filename)
                    tags = []

                    for i, tag in enumerate(meta_data['majorityvote']):
                        if tag is not 'S' and tag is not 'U':
                            tags.append(tag)
                    tags = ';'.join(tags)
                    writer.writerow(
                        (os.path.join(relative_path, raw_filename), scene_label, meta_data['majorityvote'], tags))
            finally:
                f.close()
            foot()

        all_folds_found = False
        #for fold in xrange(1, self.evaluation_folds):
        #    for target_tag in self.audio_tags:
        #        if not os.path.isfile(os.path.join(self.evaluation_setup_path,
        #                                           'fold' + str(fold) + '_' + target_tag.replace('/', '-').replace(' ',
        #                                                                                                           '_') + '_train.txt')):
        #            all_folds_found = False
        #        if not os.path.isfile(os.path.join(self.evaluation_setup_path,
        #                                           'fold' + str(fold) + '_' + target_tag.replace('/', '-').replace(' ',
        #                                                                                                           '_') + '_test.txt')):
        #            all_folds_found = False
        #        if not os.path.isfile(os.path.join(self.evaluation_setup_path,
        #                                           'fold' + str(fold) + '_' + target_tag.replace('/', '-').replace(' ',
        #                                                                                                           '_') + '_evaluate.txt')):
        #            all_folds_found = False               

        if not all_folds_found:
            if not os.path.isdir(self.evaluation_setup_path):
                os.makedirs(self.evaluation_setup_path)

            files, fold_assignments = [], []
            with open(self.package_list[0]['development_chunks_refined_crossval_csv'], 'rt') as f:
                for row in csv.reader(f, delimiter=','):
                    files.append(self.relative_to_absolute_path(os.path.join('chime_home','chunks',row[1]+'.'+self.sampling_rate+'.wav')))
                    fold_assignments.append(int(row[2]))
            files = numpy.array(files) 
            fold_assignments = numpy.array(fold_assignments)
            
            #fold_assignments = numpy.array(range(len(files))) % self.evaluation_folds
            #numpy.random.shuffle(fold_assignments)

            for fold in numpy.unique(fold_assignments): 
                train_files = files[fold_assignments!=fold]
                test_files = files[fold_assignments==fold]

                with open(os.path.join(self.evaluation_setup_path, 'fold' + str(fold+1) + '_train.txt'), 'wt') as f:
                    writer = csv.writer(f, delimiter='\t')
                    for file in train_files:
                        raw_path, raw_filename = os.path.split(file)
                        relative_path = self.absolute_to_relative(raw_path)
                        item = self.file_meta(file)[0]
                        writer.writerow([os.path.join(relative_path, raw_filename), item['scene_label'],item['tag_string'], ';'.join(item['tags'])])


                with open(os.path.join(self.evaluation_setup_path, 'fold' + str(fold+1) + '_test.txt'), 'wt') as f:
                    writer = csv.writer(f, delimiter='\t')
                    for file in test_files:
                        raw_path, raw_filename = os.path.split(file)
                        relative_path = self.absolute_to_relative(raw_path)
                        writer.writerow([os.path.join(relative_path, raw_filename)])

                #_evaluate.txt in this context refers to a list of testing files with their annotations (cf. _test.txt)
                with open(os.path.join(self.evaluation_setup_path, 'fold' + str(fold+1) + '_evaluate.txt'), 'wt') as f:
                    writer = csv.writer(f, delimiter='\t')
                    for file in test_files:
                        raw_path, raw_filename = os.path.split(file)
                        relative_path = self.absolute_to_relative(raw_path)
                        item = self.file_meta(file)[0]
                        writer.writerow([os.path.join(relative_path, raw_filename), item['scene_label'],item['tag_string'], ';'.join(item['tags'])])

class CHiMEHome_DomesticAudioTag_ChallengeSet(CHiMEHome_DomesticAudioTag_DevelopmentSet):
    def __init__(self, data_path=None, name='CHiMEHome-audiotag-challenge', evaluation_folds=1):
        CHiMEHome_DomesticAudioTag_DevelopmentSet.__init__(self, data_path, name, evaluation_folds)        
        self.package_list[0]['evaluation_chunks_refined_csv'] = os.path.join(self.local_path, 'chime_home', 'evaluation_chunks_refined.csv')                
        #This is inefficient; since the data may already have been downloaded as part of `development' mode.
        self.package_list[0]['remote_package'] = 'https://archive.org/download/chime-home/chime_home.tar.gz'
                        
    def folds(self, mode='folds'):
            return range(1, self.evaluation_folds + 1)        
        
    def on_after_extract(self):
        # Make legacy dataset compatible with DCASE2016 dataset scheme
        if not os.path.isfile(self.meta_file):
            section_header('Generating meta file for dataset')

            scene_label = 'home'
            f = open(self.meta_file, 'wt')
            try:
                writer = csv.writer(f, delimiter='\t')
                for file in self.audio_files:
                    raw_path, raw_filename = os.path.split(file)
                    relative_path = self.absolute_to_relative(raw_path)

                    base_filename, file_extension = os.path.splitext(raw_filename)
                    base_filename, sampling_rate = os.path.splitext(base_filename)
                    annotation_filename = os.path.join(raw_path, base_filename + '.csv')
                    meta_data = self.read_chunk_meta(annotation_filename)
                    tags = []

                    if meta_data is not None:
                        for i, tag in enumerate(meta_data['majorityvote']):
                            if tag is not 'S' and tag is not 'U':
                                tags.append(tag)
                        tags = ';'.join(tags)
                        writer.writerow(
                            (os.path.join(relative_path, raw_filename), scene_label, meta_data['majorityvote'], tags))
                    else:
                        writer.writerow(
                            (os.path.join(relative_path, raw_filename), None, None))
            finally:
                f.close()
            foot()

        if not os.path.isdir(self.evaluation_setup_path):
            os.makedirs(self.evaluation_setup_path)

        for fold in (1,):
            files = []
            with open(self.package_list[0]['development_chunks_refined_csv'], 'rt') as f:
                for row in csv.reader(f, delimiter=','):
                    files.append(self.relative_to_absolute_path(os.path.join('chime_home','chunks',row[1]+'.'+self.sampling_rate+'.wav')))

            with open(os.path.join(self.evaluation_setup_path, 'fold' + str(fold) + '_train.txt'), 'wt') as f:
                writer = csv.writer(f, delimiter='\t')
                for file in files:
                    raw_path, raw_filename = os.path.split(file)
                    relative_path = self.absolute_to_relative(raw_path)
                    item = self.file_meta(file)[0]
                    writer.writerow([os.path.join(relative_path, raw_filename), item['scene_label'],item['tag_string'], ';'.join(item['tags'])])

            files = []
            with open(self.package_list[0]['evaluation_chunks_refined_csv'], 'rt') as f:
                for row in csv.reader(f, delimiter=','):
                    files.append(self.relative_to_absolute_path(os.path.join('chime_home','chunks',row[1]+'.'+self.sampling_rate+'.wav')))

            with open(os.path.join(self.evaluation_setup_path, 'fold' + str(fold) + '_test.txt'), 'wt') as f:
                writer = csv.writer(f, delimiter='\t')
                for file in files:
                    raw_path, raw_filename = os.path.split(file)
                    relative_path = self.absolute_to_relative(raw_path)
                    writer.writerow([os.path.join(relative_path, raw_filename)])
