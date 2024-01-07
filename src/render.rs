use bevy::ecs::system::Resource;
use std::sync::Arc;

use vulkano::{
	buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
	command_buffer::{
		allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
		RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
	},
	command_buffer::{
		allocator::StandardCommandBufferAllocatorCreateInfo, CommandBufferInheritanceInfo,
		SecondaryAutoCommandBuffer,
	},
	descriptor_set::allocator::StandardDescriptorSetAllocator,
	device::{DeviceOwned, Queue},
	format::Format,
	image::view::ImageView,
	memory::allocator::StandardMemoryAllocator,
	memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
	pipeline::{
		graphics::{
			color_blend::{ColorBlendAttachmentState, ColorBlendState},
			input_assembly::InputAssemblyState,
			multisample::MultisampleState,
			rasterization::RasterizationState,
			vertex_input::{Vertex, VertexDefinition},
			viewport::{Viewport, ViewportState},
			GraphicsPipelineCreateInfo,
		},
		layout::PipelineDescriptorSetLayoutCreateInfo,
		DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
	},
	render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
	sync::GpuFuture,
};

#[derive(Resource)]
pub struct Render {
	gfx_queue: Arc<Queue>,
	command_buffer_allocator: StandardCommandBufferAllocator,
	render_pass: Arc<RenderPass>,
	triangle_draw_pipeline: TriangleDrawPipeline,
}

impl Render {
	pub fn new(
		allocator: Arc<StandardMemoryAllocator>,
		gfx_queue: Arc<Queue>,
		output_format: Format,
	) -> Self {
		let render_pass = vulkano::single_pass_renderpass!(gfx_queue.device().clone(),
			attachments: {
				color: {
					format: output_format,
					samples: 1,
					load_op: Clear,
					store_op: Store,
				}
			},
			pass: {
					color: [color],
					depth_stencil: {}
			}
		)
		.unwrap();
		let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

		let triangle_draw_pipeline =
			TriangleDrawPipeline::new(allocator.clone(), gfx_queue.clone(), subpass);

		Self {
			gfx_queue,
			command_buffer_allocator: StandardCommandBufferAllocator::new(
				allocator.device().clone(),
				Default::default(),
			),
			render_pass,
			triangle_draw_pipeline,
		}
	}

	pub fn render<F>(&mut self, before_future: F, target: Arc<ImageView>) -> Box<dyn GpuFuture>
	where
		F: GpuFuture + 'static,
	{
		let img_dims = target.image().extent();
		let framebuffer = Framebuffer::new(
			self.render_pass.clone(),
			FramebufferCreateInfo {
				attachments: vec![target],
				..Default::default()
			},
		)
		.unwrap();
		let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
			&self.command_buffer_allocator,
			self.gfx_queue.queue_family_index(),
			CommandBufferUsage::OneTimeSubmit,
		)
		.unwrap();
		command_buffer_builder
			.begin_render_pass(
				RenderPassBeginInfo {
					clear_values: vec![Some([0.0; 4].into())],
					..RenderPassBeginInfo::framebuffer(framebuffer)
				},
				SubpassBeginInfo {
					contents: SubpassContents::SecondaryCommandBuffers,
					..Default::default()
				},
			)
			.unwrap();
		let cb = self.triangle_draw_pipeline.draw([img_dims[0], img_dims[1]]);
		command_buffer_builder.execute_commands(cb).unwrap();
		command_buffer_builder
			.end_render_pass(Default::default())
			.unwrap();
		let command_buffer = command_buffer_builder.build().unwrap();
		let after_future = before_future
			.then_execute(self.gfx_queue.clone(), command_buffer)
			.unwrap();

		after_future.boxed()
	}
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct PosVertex {
	#[format(R32G32_SFLOAT)]
	position: [f32; 2],
}

impl PosVertex {
	pub fn new(x: f32, y: f32) -> Self {
		Self { position: [x, y] }
	}
}

fn triangle() -> Vec<PosVertex> {
	vec![
		PosVertex::new(0.0, -0.5),
		PosVertex::new(0.5, 0.5),
		PosVertex::new(-0.5, 0.5),
	]
}

pub struct TriangleDrawPipeline {
	gfx_queue: Arc<Queue>,
	command_buffer_allocator: StandardCommandBufferAllocator,
	descriptor_set_allocator: StandardDescriptorSetAllocator,
	pipeline: Arc<GraphicsPipeline>,
	subpass: Subpass,
	vertices: Subbuffer<[PosVertex]>,
}

impl TriangleDrawPipeline {
	pub fn new(
		allocator: Arc<StandardMemoryAllocator>,
		gfx_queue: Arc<Queue>,
		subpass: Subpass,
	) -> Self {
		let vertices = triangle();
		let vertex_buffer = Buffer::from_iter(
			allocator.clone(),
			BufferCreateInfo {
				usage: BufferUsage::VERTEX_BUFFER,
				..Default::default()
			},
			AllocationCreateInfo {
				memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
					| MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
				..Default::default()
			},
			vertices,
		)
		.unwrap();

		let pipeline = {
			let vs = vs::load(allocator.device().clone())
				.expect("failed to create shader module")
				.entry_point("main")
				.expect("shader entry point not found");
			let fs = fs::load(allocator.device().clone())
				.expect("failed to create shader module")
				.entry_point("main")
				.expect("shader entry point not found");
			let vertex_input_state = PosVertex::per_vertex()
				.definition(&vs.info().input_interface)
				.unwrap();
			let stages = [
				PipelineShaderStageCreateInfo::new(vs),
				PipelineShaderStageCreateInfo::new(fs),
			];
			let layout = PipelineLayout::new(
				allocator.device().clone(),
				PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
					.into_pipeline_layout_create_info(allocator.device().clone())
					.unwrap(),
			)
			.unwrap();

			GraphicsPipeline::new(
				allocator.device().clone(),
				None,
				GraphicsPipelineCreateInfo {
					stages: stages.into_iter().collect(),
					vertex_input_state: Some(vertex_input_state),
					input_assembly_state: Some(InputAssemblyState::default()),
					viewport_state: Some(ViewportState::default()),
					rasterization_state: Some(RasterizationState::default()),
					multisample_state: Some(MultisampleState::default()),
					color_blend_state: Some(ColorBlendState::with_attachment_states(
						subpass.num_color_attachments(),
						ColorBlendAttachmentState::default(),
					)),
					dynamic_state: [DynamicState::Viewport].into_iter().collect(),
					subpass: Some(subpass.clone().into()),
					..GraphicsPipelineCreateInfo::layout(layout)
				},
			)
			.unwrap()
		};
		let command_buffer_allocator = StandardCommandBufferAllocator::new(
			allocator.device().clone(),
			StandardCommandBufferAllocatorCreateInfo {
				secondary_buffer_count: 32,
				..Default::default()
			},
		);
		let descriptor_set_allocator =
			StandardDescriptorSetAllocator::new(allocator.device().clone(), Default::default());

		Self {
			gfx_queue,
			command_buffer_allocator,
			descriptor_set_allocator,
			pipeline,
			subpass,
			vertices: vertex_buffer,
		}
	}

	pub fn draw(&mut self, viewport_dimensions: [u32; 2]) -> Arc<SecondaryAutoCommandBuffer> {
		let mut builder = AutoCommandBufferBuilder::secondary(
			&self.command_buffer_allocator,
			self.gfx_queue.queue_family_index(),
			CommandBufferUsage::MultipleSubmit,
			CommandBufferInheritanceInfo {
				render_pass: Some(self.subpass.clone().into()),
				..Default::default()
			},
		)
		.unwrap();

		builder
			.set_viewport(
				0,
				[Viewport {
					offset: [0.0, 0.0],
					extent: [viewport_dimensions[0] as f32, viewport_dimensions[1] as f32],
					depth_range: 0.0..=1.0,
				}]
				.into_iter()
				.collect(),
			)
			.unwrap()
			.bind_pipeline_graphics(self.pipeline.clone())
			.unwrap()
			.bind_vertex_buffers(0, self.vertices.clone())
			.unwrap()
			.draw(self.vertices.len() as u32, 1, 0, 0)
			.unwrap();
		builder.build().unwrap()
	}
}

mod vs {
	vulkano_shaders::shader! {
		ty: "vertex",
		src: r#"
#version 460
layout (location = 0) in vec2 position;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}
"#
	}
}

mod fs {
	vulkano_shaders::shader! {
		ty: "fragment",
		src: r#"
#version 460
layout (location = 0) out vec4 f_color;

void main() {
    f_color = vec4(1.0, 0.0, 0.0, 1.0);
}
"#
	}
}
