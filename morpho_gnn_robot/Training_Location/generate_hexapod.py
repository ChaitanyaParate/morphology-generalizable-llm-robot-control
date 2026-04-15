import xml.etree.ElementTree as ET
import copy

input_urdf = "/mnt/newvolume/Programming/Python/Deep_Learning/Relational_Bias_for_Morphological_Generalization/morpho_gnn_robot/Training_Location/anymal_stripped.urdf"
output_urdf = "/mnt/newvolume/Programming/Python/Deep_Learning/Relational_Bias_for_Morphological_Generalization/morpho_gnn_robot/Training_Location/hexapod_anymal.urdf"

tree = ET.parse(input_urdf)
root = tree.getroot()

def clone_and_rename_prefix(parent, original_prefix, new_prefix, origin_x_offset=0.0):
    new_elements = []
    # Find all links and joints starting with original_prefix
    for elem in list(root):
        if 'name' in elem.attrib and elem.attrib['name'].startswith(original_prefix):
            new_elem = copy.deepcopy(elem)
            # Rename element
            new_elem.attrib['name'] = new_elem.attrib['name'].replace(original_prefix, new_prefix)
            
            # If it's a joint, update child and parent references
            if new_elem.tag == 'joint':
                for child in new_elem:
                    if child.tag == 'child':
                        child.attrib['link'] = child.attrib['link'].replace(original_prefix, new_prefix)
                    elif child.tag == 'parent' and child.attrib['link'].startswith(original_prefix):
                        child.attrib['link'] = child.attrib['link'].replace(original_prefix, new_prefix)
                    elif child.tag == 'origin' and 'xyz' in child.attrib:
                        # Only modify the origin of the HAA joint that connects to the base
                        if new_elem.attrib['name'].endswith('_HAA'):
                            xyz = child.attrib['xyz'].split()
                            xyz[0] = str(float(xyz[0]) + origin_x_offset)  # Shift X
                            child.attrib['xyz'] = ' '.join(xyz)
                            
            # Add to list
            new_elements.append(new_elem)
            
    # Also copy gazebo reference plugins
    for elem in list(root):
        if elem.tag == 'gazebo' and 'reference' in elem.attrib and elem.attrib['reference'].startswith(original_prefix):
            new_elem = copy.deepcopy(elem)
            new_elem.attrib['reference'] = new_elem.attrib['reference'].replace(original_prefix, new_prefix)
            new_elements.append(new_elem)
            
    # Also find gazebo plugin systems referencing the original_prefix joints
    for gazebo in root.findall('gazebo'):
        for plugin in gazebo.findall('plugin'):
            for joint_name in plugin.findall('joint_name'):
                if joint_name.text and joint_name.text.startswith(original_prefix):
                    new_plugin = copy.deepcopy(plugin)
                    new_plugin.find('joint_name').text = new_plugin.find('joint_name').text.replace(original_prefix, new_prefix)
                    topic = new_plugin.find('topic')
                    if topic is not None and topic.text:
                        topic.text = topic.text.replace(original_prefix, new_prefix)
                    gazebo.append(new_plugin)
                    
    # Append the newly cloned elements to root
    for ne in new_elements:
        root.append(ne)

# Clone LF (Left Front) -> LM (Left Middle)
# LF_HAA xyz origin x is 0.277. We want LM to be at x=0.0, so offset is -0.277
clone_and_rename_prefix(root, 'LF_', 'LM_', origin_x_offset=-0.277)

# Clone RF (Right Front) -> RM (Right Middle)
# RF_HAA xyz origin x is 0.277. We want RM to be at x=0.0, so offset is -0.277
clone_and_rename_prefix(root, 'RF_', 'RM_', origin_x_offset=-0.277)

with open(output_urdf, 'wb') as f:
    # Adding xml declaration manually for cleanliness
    f.write(b'<?xml version="1.0" ?>\n')
    tree.write(f, encoding='utf-8')

print(f"Generated {output_urdf} with 18 joints successfully!")
