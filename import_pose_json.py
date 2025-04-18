import bpy
import json
import os
from mathutils import Vector, Matrix
import math

# --- 配置 ---
# 尺寸调整：缩小为原来的 2/3
JOINT_RADIUS_MAIN = 5.0 * (2.0 / 3.0) # 手臂/腿部关节球体的半径
JOINT_RADIUS_DETAIL = (5.0 / 3.0) * (2.0 / 3.0) # 手/脚 关节球体半径
FACE_RADIUS_FACTOR = 3.0 # 脸部球体半径相对于细节关节半径的倍数
POSITION_SCALE = 100.0 # 坐标放大倍数
BONE_RADIUS_BODY = 1.5 # 身体主要骨骼圆柱体的半径
BONE_RADIUS_DETAIL = 0.5 # 手、脚、脸部骨骼使用更细的半径
CREATE_JOINTS = True # 是否创建（剩余的）关节球体
CREATE_BONES = True
REPLACE_FACE_WITH_SPHERE = True
# DEBUG_TORSO_VERTICES = True # 不再需要打印，用 Shape Keys 动画

# --- 颜色和材质定义 ---
def create_material(name, color):
    """创建或获取一个具有指定漫反射颜色的材质。"""
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        principled_bsdf = mat.node_tree.nodes.get('Principled BSDF')
        if principled_bsdf:
            # 设置材质节点颜色 (用于渲染)
            principled_bsdf.inputs['Base Color'].default_value = color
    # 也设置材质的基础漫反射颜色 (有时在视窗中更直接)
    mat.diffuse_color = color
    return mat

def assign_material_and_color(obj, material, color_rgba):
    """安全地分配材质并设置对象颜色。"""
    if obj and obj.data:
        if material:
            # 清空现有材质槽并添加新材质
            obj.data.materials.clear()
            obj.data.materials.append(material)
        # 设置对象颜色 (对 Solid Mode 视窗显示有效)
        obj.color = color_rgba[:4] # 确保是4元组 RGBA

# 颜色定义 (不变)
COLOR_HEAD_RED = (1.0, 0.0, 0.0, 1.0)  # 红色
COLOR_TORSO_GREEN = (0.0, 1.0, 0.0, 1.0) # 绿色
COLOR_ARM_YELLOW = (1.0, 1.0, 0.0, 1.0) # 黄色
COLOR_LEG_BLUE = (0.0, 0.0, 1.0, 1.0) # 蓝色
COLOR_HAND_FOOT_PURPLE = (0.5, 0.0, 0.5, 1.0) # 紫色
COLOR_BONE = (0.7, 0.7, 0.7, 1.0) # 灰色

# 材质字典 (不变)
materials = {
    'head': create_material("Mat_Head_Red", COLOR_HEAD_RED),
    'torso': create_material("Mat_Torso_Green", COLOR_TORSO_GREEN),
    'arm': create_material("Mat_Arm_Yellow", COLOR_ARM_YELLOW),
    'leg': create_material("Mat_Leg_Blue", COLOR_LEG_BLUE),
    'hand_foot': create_material("Mat_HandFoot_Purple", COLOR_HAND_FOOT_PURPLE),
    'bone': create_material("Mat_Bone", COLOR_BONE)
}

# 颜色字典 (方便直接获取颜色)
colors = {
    'head': COLOR_HEAD_RED,
    'torso': COLOR_TORSO_GREEN,
    'arm': COLOR_ARM_YELLOW,
    'leg': COLOR_LEG_BLUE,
    'hand_foot': COLOR_HAND_FOOT_PURPLE,
    'bone': COLOR_BONE
}

# --- 关键点定义 (不变) ---
FACE_LANDMARKS = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT']
TORSO_LANDMARKS = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP']
FACE_CENTER_CALC_LANDMARKS = ['LEFT_EAR', 'RIGHT_EAR', 'NOSE', 'MOUTH_LEFT', 'MOUTH_RIGHT']
FACE_RADIUS_CALC_LANDMARKS = FACE_LANDMARKS
HAND_FOOT_LANDMARKS = [
    'LEFT_WRIST', 'LEFT_PINKY', 'LEFT_INDEX', 'LEFT_THUMB',
    'RIGHT_WRIST', 'RIGHT_PINKY', 'RIGHT_INDEX', 'RIGHT_THUMB',
    'LEFT_ANKLE', 'LEFT_HEEL', 'LEFT_FOOT_INDEX',
    'RIGHT_ANKLE', 'RIGHT_HEEL', 'RIGHT_FOOT_INDEX'
]
ARM_LANDMARKS = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW']
LEG_LANDMARKS = ['LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE']

# --- 关键点分组获取函数 (不变) ---
def get_landmark_group(landmark_name):
    if landmark_name in FACE_LANDMARKS: return 'head'
    if landmark_name in TORSO_LANDMARKS: return 'torso'
    if landmark_name in HAND_FOOT_LANDMARKS: return 'hand_foot'
    if landmark_name in ARM_LANDMARKS: return 'arm'
    if landmark_name in LEG_LANDMARKS: return 'leg'
    print(f"警告：关键点 '{landmark_name}' 未找到明确分组，默认指定为 'arm'")
    return 'arm'

# --- 定义骨骼连接 (移除耳朵-肩膀连接) ---
def is_detail_bone(p1_name, p2_name):
    """判断连接是否属于手、脚或脸部的细节连接"""
    detail_landmarks = HAND_FOOT_LANDMARKS + FACE_LANDMARKS
    is_p1_detail = p1_name in detail_landmarks
    is_p2_detail = p2_name in detail_landmarks
    return is_p1_detail and is_p2_detail

# 原始连接，后续会根据替换选项过滤
ALL_POSE_CONNECTIONS = [
    # 恢复躯干内部连接
    ('LEFT_SHOULDER', 'RIGHT_SHOULDER'), ('LEFT_HIP', 'RIGHT_HIP'),
    ('LEFT_SHOULDER', 'LEFT_HIP'), ('RIGHT_SHOULDER', 'RIGHT_HIP'),
    # 脸部内部 (会被替换, 保留定义用于过滤)
    ('LEFT_EYE_INNER', 'LEFT_EYE'), ('LEFT_EYE', 'LEFT_EYE_OUTER'),
    ('RIGHT_EYE_INNER', 'RIGHT_EYE'), ('RIGHT_EYE', 'RIGHT_EYE_OUTER'),
    ('LEFT_EAR', 'LEFT_EYE_OUTER'), ('RIGHT_EAR', 'RIGHT_EYE_OUTER'),
    ('NOSE', 'LEFT_EYE_INNER'), ('NOSE', 'RIGHT_EYE_INNER'),
    ('MOUTH_LEFT', 'MOUTH_RIGHT'),
    ('LEFT_EYE_OUTER', 'MOUTH_LEFT'), ('RIGHT_EYE_OUTER', 'MOUTH_RIGHT'),
    ('NOSE', 'MOUTH_LEFT'), ('NOSE', 'MOUTH_RIGHT'),
    # 手臂
    ('LEFT_SHOULDER', 'LEFT_ELBOW'), ('LEFT_ELBOW', 'LEFT_WRIST'),
    ('RIGHT_SHOULDER', 'RIGHT_ELBOW'), ('RIGHT_ELBOW', 'RIGHT_WRIST'),
    # 腿部
    ('LEFT_HIP', 'LEFT_KNEE'), ('LEFT_KNEE', 'LEFT_ANKLE'),
    ('RIGHT_HIP', 'RIGHT_KNEE'), ('RIGHT_KNEE', 'RIGHT_ANKLE'),
    # 手指
    ('LEFT_WRIST', 'LEFT_PINKY'), ('LEFT_WRIST', 'LEFT_INDEX'), ('LEFT_WRIST', 'LEFT_THUMB'),
    ('RIGHT_WRIST', 'RIGHT_PINKY'), ('RIGHT_WRIST', 'RIGHT_INDEX'), ('RIGHT_WRIST', 'RIGHT_THUMB'),
    # 脚部
    ('LEFT_ANKLE', 'LEFT_HEEL'), ('LEFT_ANKLE', 'LEFT_FOOT_INDEX'),
    ('RIGHT_ANKLE', 'RIGHT_HEEL'), ('RIGHT_ANKLE', 'RIGHT_FOOT_INDEX'),
    # 连接头和躯干 (移除)
    # ('LEFT_EAR', 'LEFT_SHOULDER'),
    # ('RIGHT_EAR', 'RIGHT_SHOULDER'),
]

# --- Blender Operator --- (execute 方法再次调整)
class ImportPoseJSONOperator(bpy.types.Operator):
    """从 JSON 文件导入 Mediapipe 姿态动画"""
    bl_idname = "scene.import_pose_json"
    bl_label = "Import Pose JSON"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(
        subtype="FILE_PATH",
        name="JSON File",
        description="选择 pose_data.json 文件"
    )
    filter_glob: bpy.props.StringProperty(default='*.json', options={'HIDDEN'})

    def execute(self, context):
        """执行导入操作。"""
        print(f"--- 开始执行导入脚本 (文件: {self.filepath}) ---")
        
        scene = context.scene
        root_collection = scene.collection

        # --- 读取全局配置到局部变量 ---
        local_create_joints = CREATE_JOINTS
        local_create_bones = CREATE_BONES
        local_replace_face = REPLACE_FACE_WITH_SPHERE

        # --- 清理旧对象 --- (修改为也清理 Face 和 Torso 对象)
        print("清理旧的导入对象...")
        collections_to_clear = ["PoseJoints", "PoseBones", "PoseFace"]
        for coll_name in collections_to_clear:
            old_coll = bpy.data.collections.get(coll_name)
            if old_coll:
                bpy.ops.object.select_all(action='DESELECT')
                objects_to_delete = [obj for obj in old_coll.objects if obj]
                if objects_to_delete:
                    for obj in objects_to_delete:
                        obj.select_set(True)
                    bpy.ops.object.delete()
                try:
                    bpy.data.collections.remove(old_coll)
                except Exception as e_coll:
                    print(f"移除集合 {coll_name} 时出错: {e_coll}")
        print("清理完成。")

        # --- 加载数据 ---
        print(f"尝试从 '{self.filepath}' 加载姿态数据...")
        if not os.path.exists(self.filepath):
            msg = f"错误: JSON 文件不存在: {self.filepath}"
            print(msg)
            self.report({'ERROR'}, msg) # 在 Blender UI 报告错误
            return {'CANCELLED'}

        try:
            with open(self.filepath, 'r') as f:
                all_frames_data = json.load(f)
        except Exception as e:
            msg = f"错误: 加载或解析 JSON 文件时出错: {e}"
            print(msg)
            self.report({'ERROR'}, msg)
            return {'CANCELLED'}

        if not all_frames_data:
            msg = "错误: JSON 文件为空或数据无效。"
            print(msg)
            self.report({'ERROR'}, msg)
            return {'CANCELLED'}

        print(f"成功加载 {len(all_frames_data)} 帧数据。")

        # --- 准备工作 ---
        scene.frame_end = len(all_frames_data) - 1
        first_frame_data = all_frames_data[0] # 用于计算初始脸部/躯干

        # --- 创建集合 ---
        joint_collection = bpy.data.collections.new("PoseJoints")
        root_collection.children.link(joint_collection)
        if local_create_bones:
            bone_collection = bpy.data.collections.new("PoseBones")
            root_collection.children.link(bone_collection)
        if local_replace_face:
            face_collection = bpy.data.collections.new("PoseFace")
            root_collection.children.link(face_collection)

        # --- 数据转换和初始计算 ---
        # 将第一帧的坐标从 list 转为 Vector 并缩放
        first_frame_vecs = {}
        for name, coords_list in first_frame_data.items():
            blender_x = -coords_list[0] * POSITION_SCALE
            blender_y = -coords_list[2] * POSITION_SCALE
            blender_z = -coords_list[1] * POSITION_SCALE
            first_frame_vecs[name] = Vector((blender_x, blender_y, blender_z))

        # --- 创建脸部替代球体 (使用 assign_material_and_color) ---
        face_sphere_obj = None
        if local_replace_face:
            print("创建脸部替代球体 (红色)...")
            center_points = [first_frame_vecs[name] for name in FACE_CENTER_CALC_LANDMARKS if name in first_frame_vecs]
            face_center = sum(center_points, Vector()) / len(center_points) if center_points else Vector((0,0,0))
            face_radius = JOINT_RADIUS_DETAIL * FACE_RADIUS_FACTOR
            bpy.ops.mesh.primitive_uv_sphere_add(radius=face_radius, segments=32, ring_count=16, location=face_center)
            face_sphere_obj = context.object
            face_sphere_obj.name = "FaceSphere"
            bpy.ops.object.shade_smooth()
            assign_material_and_color(face_sphere_obj, materials.get('head'), colors.get('head'))
            try: root_collection.objects.unlink(face_sphere_obj)
            except: pass
            face_collection.objects.link(face_sphere_obj)

        # --- 创建（剩余的）关节球体 (区分半径, 使用 assign_material_and_color) ---
        joint_objects = {}
        if local_create_joints:
            print("创建剩余的关节球体 (黄/蓝/紫)...")
            for name in first_frame_data.keys():
                # 如果启用了脸部替换且当前点是脸部点，则跳过
                if local_replace_face and name in FACE_LANDMARKS:
                    continue

                # 根据分组选择半径
                group = get_landmark_group(name)
                current_joint_radius = JOINT_RADIUS_DETAIL if group == 'hand_foot' else JOINT_RADIUS_MAIN

                bpy.ops.mesh.primitive_uv_sphere_add(radius=current_joint_radius, segments=16, ring_count=8, location=first_frame_vecs.get(name, (0,0,0)))
                joint_obj = context.object
                joint_obj.name = f"Joint_{name}"
                joint_objects[name] = joint_obj
                bpy.ops.object.shade_smooth()
                assign_material_and_color(joint_obj, materials.get(group), colors.get(group))
                try: root_collection.objects.unlink(joint_obj)
                except: pass
                joint_collection.objects.link(joint_obj)

        # --- 创建（需要的）骨骼对象 (使用 assign_material_and_color) ---
        bone_objects = {}
        if local_create_bones:
            print("创建骨骼对象...")
            for p1_name, p2_name in ALL_POSE_CONNECTIONS: # 使用更新后的列表
                create_this_bone = True
                if local_replace_face and p1_name in FACE_LANDMARKS and p2_name in FACE_LANDMARKS: create_this_bone = False

                p1_exists = (local_replace_face and p1_name in FACE_LANDMARKS) or (p1_name in joint_objects)
                p2_exists = (local_replace_face and p2_name in FACE_LANDMARKS) or (p2_name in joint_objects)

                if create_this_bone and p1_exists and p2_exists:
                    # filtered_connections.append((p1_name, p2_name)) # 不再需要
                    current_bone_radius = BONE_RADIUS_DETAIL if is_detail_bone(p1_name, p2_name) else BONE_RADIUS_BODY
                    bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=current_bone_radius, depth=1, location=(0, 0, 0))
                    bone_obj = context.object
                    bone_obj.name = f"Bone_{p1_name}_{p2_name}"
                    bone_objects[(p1_name, p2_name)] = bone_obj
                    bpy.ops.object.shade_smooth()
                    assign_material_and_color(bone_obj, materials.get('bone'), colors.get('bone'))
                    try: root_collection.objects.unlink(bone_obj)
                    except: pass
                    bone_collection.objects.link(bone_obj)

        # --- 设置动画关键帧 (添加地面固定逻辑) ---
        print("设置动画关键帧...")
        foot_landmark_names = ['LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']

        for frame_index, frame_data in enumerate(all_frames_data):
            scene.frame_set(frame_index)
            if frame_index % 50 == 0 or frame_index == len(all_frames_data) - 1:
                print(f"处理帧: {frame_index}")

            if not frame_data or len(frame_data) != 33:
                 print(f"警告：跳过帧 {frame_index}，数据缺失。")
                 continue

            # --- 转换当前帧坐标 (原始) ---
            raw_frame_vecs = {}
            for name, coords_list in frame_data.items():
                blender_x = -coords_list[0] * POSITION_SCALE
                blender_y = -coords_list[2] * POSITION_SCALE
                blender_z = -coords_list[1] * POSITION_SCALE
                raw_frame_vecs[name] = Vector((blender_x, blender_y, blender_z))

            # --- 计算地面固定偏移 --- 
            min_z = float('inf')
            foot_points_found = False
            for name in foot_landmark_names:
                if name in raw_frame_vecs:
                    min_z = min(min_z, raw_frame_vecs[name].z)
                    foot_points_found = True
            
            z_shift = 0 # 默认不偏移
            if foot_points_found and min_z < float('inf'):
                 z_shift = -min_z # 计算向上移动的距离
                 # print(f"Frame {frame_index}: min_z={min_z:.3f}, z_shift={z_shift:.3f}") # 可选调试
            # else:
                 # print(f"Frame {frame_index}: No foot points found, no z-shift applied.") # 可选调试
                 
            shift_vector = Vector((0, 0, z_shift))
            
            # --- 应用偏移到所有点 --- 
            current_frame_vecs = {name: vec + shift_vector for name, vec in raw_frame_vecs.items()}

            # --- 更新脸部球体 (使用偏移后的坐标) ---
            if local_replace_face and face_sphere_obj:
                center_points = [current_frame_vecs[name] for name in FACE_CENTER_CALC_LANDMARKS if name in current_frame_vecs]
                if center_points: # 确保列表不为空
                    face_center = sum(center_points, Vector()) / len(center_points)
                    face_sphere_obj.location = face_center
                    face_sphere_obj.keyframe_insert(data_path="location", frame=frame_index)
                # else: # 如果计算中心的点缺失，保持上一帧位置？或者用偏移后的鼻子？
                    # nose_loc = current_frame_vecs.get('NOSE')
                    # if nose_loc: face_sphere_obj.location = nose_loc ...
                    # pass # 暂时不做处理

            # --- 更新剩余关节球体位置 ---
            if local_create_joints:
                for name, obj in joint_objects.items():
                    if name in current_frame_vecs:
                        obj.location = current_frame_vecs[name]
                        obj.keyframe_insert(data_path="location", frame=frame_index)

            # --- 更新需要的骨骼 ---
            if local_create_bones:
                 for (p1_name, p2_name), bone_obj in bone_objects.items():
                     # 获取偏移后的端点位置
                     loc1 = face_sphere_obj.location if (local_replace_face and p1_name in FACE_LANDMARKS) else current_frame_vecs.get(p1_name)
                     loc2 = face_sphere_obj.location if (local_replace_face and p2_name in FACE_LANDMARKS) else current_frame_vecs.get(p2_name)

                     if loc1 and loc2:
                         bone_obj.location = (loc1 + loc2) / 2.0
                         diff = loc2 - loc1
                         length = diff.length
                         if length > 1e-6:
                             quat = diff.to_track_quat('Z', 'Y')
                             bone_obj.rotation_euler = quat.to_euler('XYZ')
                             bone_obj.scale = (1.0, 1.0, length)
                         else:
                             bone_obj.scale = (1.0, 1.0, 0.0)
                         bone_obj.keyframe_insert(data_path="location", frame=frame_index)
                         bone_obj.keyframe_insert(data_path="rotation_euler", frame=frame_index)
                         bone_obj.keyframe_insert(data_path="scale", frame=frame_index)

        print("动画关键帧设置完毕！")
        self.report({'INFO'}, f"成功导入 {len(all_frames_data)} 帧姿态动画。")
        return {'FINISHED'}

    def invoke(self, context, event):
        """打开文件选择器。"""
        # 尝试从环境变量获取路径作为默认值
        default_path = os.environ.get('POSE_JSON_FILE', "")
        if not default_path or not os.path.exists(default_path):
             # 如果环境变量无效，尝试基于脚本位置猜测
             script_dir = os.path.dirname(__file__) if "__file__" in locals() else "."
             potential_path = os.path.join(script_dir, "pose_data.json")
             if os.path.exists(potential_path):
                  self.filepath = potential_path
             else:
                  self.filepath = "pose_data.json" # 默认文件名
        else:
            self.filepath = default_path # 使用环境变量中的路径
            
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'} # 返回 MODAL 状态，让文件选择器保持打开

# --- 注册和运行 Operator ---
def register():
    bpy.utils.register_class(ImportPoseJSONOperator)

def unregister():
    bpy.utils.unregister_class(ImportPoseJSONOperator)

# 主执行块: 使得脚本可以直接在 Blender 文本编辑器中运行
if __name__ == "__main__":
    # 先注销，防止重复注册报错
    try:
        unregister()
    except Exception:
        pass
    # 注册 Operator
    register()
    
    # 检查是否通过命令行 -P 运行 (环境变量应该存在)
    json_file_from_env = os.environ.get('POSE_JSON_FILE')
    
    if json_file_from_env and os.path.exists(json_file_from_env):
        print(f"检测到环境变量 POSE_JSON_FILE，自动执行导入: {json_file_from_env}")
        # 直接调用 execute，传入文件路径
        bpy.ops.scene.import_pose_json('EXEC_DEFAULT', filepath=json_file_from_env)
    else:
        print("未检测到有效的 POSE_JSON_FILE 环境变量，将打开文件选择器。")
        # 调用 invoke 来打开文件选择器
        bpy.ops.scene.import_pose_json('INVOKE_DEFAULT') 