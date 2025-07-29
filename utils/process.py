import os
def rename_files_with_character(src_dir, character):

    for root, dirs, files in os.walk(src_dir):
        for filename in files:
            if filename.endswith('.jpg'):
                src_file_path = os.path.join(root, filename)
                

                basename, extension = os.path.splitext(filename)


                parent_folder_name = os.path.basename(root)
                
                if 'D' in parent_folder_name:
                    new_prefix = 'A'
                elif 'H' in parent_folder_name:
                    new_prefix = 'B'
                else:
                    new_prefix = ''
                new_basename = basename.replace(character, new_prefix)

                new_filename = new_basename + extension
                

                dest_file_path = os.path.join(root, new_filename)
                

                os.rename(src_file_path, dest_file_path)
                print(f"Renamed {src_file_path} to {dest_file_path}")

src_folder = ''

character_to_replace = 'B'

rename_files_with_character(src_folder, character_to_replace) 
