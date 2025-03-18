from unstructured.partition.pdf import partition_pdf
import os
filename = os.path.abspath("content/cj.pdf")
print(filename)

# raw_pdf_elements=partition_pdf(
#     filename="/content/cj.pdf",                  # mandatory
#     strategy="hi_res",                                 # mandatory to use ``hi_res`` strategy
#     extract_images_in_pdf=True,                       # mandatory to set as ``True``
#     extract_image_block_types=["Image", "Table"],          # optional
#     extract_image_block_to_payload=False,                  # optional
#     extract_image_block_output_dir="extracted_data",  # optional - only works when ``extract_image_block_to_payload=False``
#     )